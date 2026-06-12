"""
评估微调后的 VLM 模型在多个测试集上的准确率（逐样本推理）。

支持的指标：
  - EM（Exact Match）：标准化后精确匹配，适用于选择题
  - Contain（词边界包含）：target 作为独立单词出现在预测中
  - Option-index accuracy：仅多选题，预测答案与选项的 fuzzy 匹配
  - ROUGE-L F1：最长公共子序列 F1 分数，适用于开放题答案
  - BERTScore F1：语义相似度，适用于推理链评估

使用示例：
  python scripts/eval.py
  python scripts/eval.py --jsonl data/processed/scienceqa_validation.jsonl data/processed/mathvista_validation.jsonl
  python scripts/eval.py --test-mode no_options -v
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import yaml   # pip install pyyaml

import torch
from qwen_vl_utils import process_vision_info

from src.eval import (
    build_prompt,
    compute_bertscore,
    compute_rouge_l,
    extract_answer,
    extract_reasoning,
    is_contained,
    is_exact_match,
    best_option_match,
)
from src.data import load_jsonl_files
from src.model import load_model_and_processor, resolve_lora_path


# ============================================================================
# 主评估流程
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估微调后的 VLM 在多测试集上的准确率"
    )

    # ---- 模型和数据 ----
    parser.add_argument("--model", default="/mnt/workspace/vlm-model-dir",
                        help="基础模型路径")
    parser.add_argument("--jsonl", nargs="+",
                        default=[
                            "data/processed/scienceqa_validation.jsonl",
                            "data/processed/mathvista_validation.jsonl",
                        ],
                        help="一个或多个测试 JSONL 文件")
    parser.add_argument("--lora", default="outputs/checkpoints/qlora_multitask",
                        help="LoRA adapter 路径（目录或具体 checkpoint）")

    # ---- 评估模式 ----
    parser.add_argument("--test-mode", choices=["with_options", "no_options"],
                        default="with_options",
                        help="评估时是否提供候选选项")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="每个文件最多评估多少条（0 = 全部）")

    # ---- 生成参数 ----
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)

    # ---- 输出控制 ----
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印每一个非 EM 样本的详细信息")
    parser.add_argument("--save-results", type=str, default="",
                        help="将评估结果保存为 JSON 文件（可选）")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径（CLI 参数会覆盖配置文件中的值）")

    # ---- 两阶段参数解析：先读 YAML 配置文件，CLI 参数自动覆盖 ----
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**config)
        print(f"已加载配置文件: {config_path}")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    print("=" * 60)
    print("加载测试数据...")
    samples, file_stats = load_jsonl_files(args.jsonl)
    print(f"总计保留 {len(samples)} 条有效样本\n")

    if not samples:
        print("错误：无有效样本，退出。")
        return

    # ------------------------------------------------------------------
    # 2. 加载模型
    # ------------------------------------------------------------------
    print("加载模型...")
    lora_path = resolve_lora_path(args.lora)
    model, processor = load_model_and_processor(
        args.model, lora_path, local_files_only=True
    )
    print("  模型加载完成\n")

    # ------------------------------------------------------------------
    # 3. 逐样本评估
    # ------------------------------------------------------------------
    total = 0
    correct_em = 0
    correct_contain = 0
    correct_option = 0
    mc_total = 0
    sum_rouge_l = 0.0
    sum_bert_score = 0.0
    n_rationale = 0

    per_dataset = defaultdict(lambda: {
        "total": 0, "em": 0, "contain": 0, "option": 0,
        "mc": 0, "rouge_sum": 0.0, "bert_sum": 0.0, "n_rat": 0,
    })

    n_total = len(samples)

    for idx, sample in enumerate(samples):
        image_path = sample["image"]
        question = sample["question"]
        choices = sample["choices"]
        answer_idx = sample["answer"]
        answer_text = sample["answer_text"]
        gt_rationale = sample.get("rationale") or ""
        source = sample.get("_source", "unknown")
        ds_name = source.replace(".jsonl", "")

        # ---- 构建 prompt ----
        eval_choices = choices if args.test_mode == "with_options" else None
        prompt = build_prompt(question, eval_choices)

        # ---- 推理 ----
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        output_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        decoded = processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        pred_text = decoded[0]
        pred_answer = extract_answer(pred_text)

        # ---- 计算各项指标 ----
        em_ok = is_exact_match(pred_answer, answer_text)
        contain_ok = is_contained(pred_answer, answer_text)
        rouge_score = compute_rouge_l(pred_answer, answer_text)

        pred_reasoning = extract_reasoning(pred_text)
        bert_score = 0.0
        if gt_rationale and pred_reasoning:
            bert_score = compute_bertscore(pred_reasoning, gt_rationale)

        # ---- 累计全局 ----
        total += 1
        if em_ok:
            correct_em += 1
        if contain_ok:
            correct_contain += 1
        sum_rouge_l += rouge_score
        if gt_rationale:
            sum_bert_score += bert_score
            n_rationale += 1

        # ---- Option-index 匹配（仅多选题）----
        option_ok = False
        if choices and isinstance(answer_idx, int):
            mc_total += 1
            per_dataset[ds_name]["mc"] += 1
            matched_idx = best_option_match(pred_answer, choices)
            if matched_idx is not None and matched_idx == answer_idx:
                correct_option += 1
                per_dataset[ds_name]["option"] += 1
                option_ok = True

        # ---- 累计按数据集 ----
        per_dataset[ds_name]["total"] += 1
        if em_ok:
            per_dataset[ds_name]["em"] += 1
        if contain_ok:
            per_dataset[ds_name]["contain"] += 1
        per_dataset[ds_name]["rouge_sum"] += rouge_score
        if gt_rationale:
            per_dataset[ds_name]["bert_sum"] += bert_score
            per_dataset[ds_name]["n_rat"] += 1

        # ---- 详细日志 ----
        if args.verbose and not em_ok:
            print(f"\n{'=' * 60}")
            print(f"[{total}/{n_total}] [{ds_name}] EM=FALSE  "
                  f"contain={contain_ok}  option={option_ok}  "
                  f"ROUGE-L={rouge_score:.3f}  BERTScore={bert_score:.3f}")
            print(f"  Q: {question[:120]}")
            print(f"  GT answer: {repr(answer_text)}    GT idx: {answer_idx}")
            print(f"  Choices: {choices}")
            print(f"  Pred raw: {pred_text[:200]}")
            print(f"  Pred extracted: {repr(pred_answer)}")

        # ---- 进度 ----
        if total % 20 == 0:
            bert_avg = sum_bert_score / max(n_rationale, 1)
            print(f"[{total}/{n_total}]  "
                  f"EM={correct_em / total:.3f}  "
                  f"contain={correct_contain / total:.3f}  "
                  f"ROUGE-L={sum_rouge_l / total:.3f}  "
                  f"BERTScore={bert_avg:.3f}  "
                  f"option={correct_option / max(mc_total, 1):.3f}")

    # ------------------------------------------------------------------
    # 4. 最终报告
    # ------------------------------------------------------------------
    bert_avg = sum_bert_score / max(n_rationale, 1)
    print("\n" + "=" * 60)
    print("评估完成 — 汇总报告")
    print("=" * 60)
    print(f"  总样本数               : {total}")
    print(f"  其中多选题             : {mc_total}")
    print(f"  有推理链的样本数       : {n_rationale}")
    print(f"  EM (精确匹配)          : {correct_em}/{total} = {correct_em / total:.4f}")
    print(f"  Contain (词边界包含)   : {correct_contain}/{total} = {correct_contain / total:.4f}")
    print(f"  ROUGE-L F1 (答案)      : {sum_rouge_l / total:.4f}")
    print(f"  BERTScore  (推理链)    : {bert_avg:.4f}  (n={n_rationale})")
    if mc_total:
        print(f"  Option-index accuracy  : {correct_option}/{mc_total} = {correct_option / mc_total:.4f}")

    # ---- 按数据集分别报告 ----
    if len(per_dataset) > 1:
        print("\n" + "-" * 40)
        print("按数据集分别统计:")
        print("-" * 40)
        for ds_name in sorted(per_dataset.keys()):
            st = per_dataset[ds_name]
            n = st["total"]
            if n == 0:
                continue
            print(f"\n  [{ds_name}]  (n={n})")
            print(f"    EM          : {st['em']}/{n} = {st['em'] / n:.4f}")
            print(f"    Contain     : {st['contain']}/{n} = {st['contain'] / n:.4f}")
            print(f"    ROUGE-L F1  : {st['rouge_sum'] / n:.4f}")
            if st["n_rat"] > 0:
                print(f"    BERTScore   : {st['bert_sum'] / st['n_rat']:.4f}  (n_rat={st['n_rat']})")
            if st["mc"]:
                print(f"    Option-Index: {st['option']}/{st['mc']} = {st['option'] / st['mc']:.4f}")

    # ---- 一行摘要 ----
    print("\n=== TL;DR ===")
    print(f"EM={correct_em / total:.4f}  "
          f"contain={correct_contain / total:.4f}  "
          f"ROUGE-L={sum_rouge_l / total:.4f}  "
          f"BERTScore={bert_avg:.4f}  "
          f"option={correct_option / max(mc_total, 1):.4f}")

    # ---- 可选：保存到 JSON ----
    if args.save_results:
        result_data = {
            "total": total,
            "mc_total": mc_total,
            "n_rationale": n_rationale,
            "em": correct_em,
            "contain": correct_contain,
            "option": correct_option,
            "rouge_l_f1": sum_rouge_l / total,
            "bertscore_f1": sum_bert_score / max(n_rationale, 1),
            "per_dataset": {
                ds: {
                    "total": st["total"],
                    "em": st["em"] / max(st["total"], 1),
                    "contain": st["contain"] / max(st["total"], 1),
                    "rouge_l_f1": st["rouge_sum"] / max(st["total"], 1),
                    "bertscore_f1": st["bert_sum"] / max(st["n_rat"], 1) if st["n_rat"] else None,
                    "option": st["option"] / max(st["mc"], 1) if st["mc"] else None,
                }
                for ds, st in per_dataset.items()
            },
        }
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.save_results}")


if __name__ == "__main__":
    main()
