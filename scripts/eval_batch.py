"""
评估微调后的 VLM 模型在多个测试集上的准确率（批量推理版）。

与 eval.py 功能完全一致，但支持批量推理（--batch-size），显著提升评估速度。

支持的指标：
  - EM / Contain / Option-index / ROUGE-L / BERTScore

使用示例：
  python scripts/eval_batch.py
  python scripts/eval_batch.py --batch-size 8 --test-mode no_options -v
  python scripts/eval_batch.py --batch-size 4 --save-results results.json
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
# 批量推理辅助函数
# ============================================================================

def _build_batch_inputs(
    batch_samples: list[dict],
    processor,
    test_mode: str,
    model_device: torch.device,
) -> tuple[dict, list[str], list[str], list[bool]]:
    """为一批样本构建模型输入。

    对 batch 中每个样本分别构建 messages、应用 chat template、
    提取图像 tensor，然后合并为单次 processor 调用。

    返回:
        (inputs, texts, image_paths, error_flags)
        inputs:       可直接传给 model.generate() 的 dict
        texts:        各样本的 chat template 文本（用于调试）
        image_paths:  各样本的图像路径（用于调试日志）
        error_flags:  各样本是否有预处理错误
    """
    batch_texts = []
    batch_images = []
    batch_image_paths = []
    batch_error = []

    for sample in batch_samples:
        image_path = sample["image"]
        question = sample["question"]
        choices = sample["choices"] if test_mode == "with_options" else None
        prompt = build_prompt(question, choices)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        batch_texts.append(text)
        batch_image_paths.append(image_path)

        if image_inputs:
            batch_images.append(image_inputs[0])
            batch_error.append(False)
        else:
            batch_images.append(torch.zeros(1))
            batch_error.append(True)

    if not batch_texts:
        return {}, batch_texts, batch_image_paths, batch_error

    inputs = processor(
        text=batch_texts,
        images=batch_images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(model_device)

    return inputs, batch_texts, batch_image_paths, batch_error


def _evaluate_single_sample(
    pred_text: str,
    sample: dict,
    error_flag: bool,
    args: argparse.Namespace,
    counters: dict,
    per_dataset: dict,
    global_idx: int,
    n_total: int,
) -> None:
    """对单个解码后的样本计算所有指标并更新计数器。"""
    if error_flag:
        return

    answer_text = sample["answer_text"]
    answer_idx = sample["answer"]
    choices = sample["choices"]
    gt_rationale = sample.get("rationale") or ""
    source = sample.get("_source", "unknown")
    ds_name = source.replace(".jsonl", "")

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
    counters["total"] += 1
    if em_ok:
        counters["correct_em"] += 1
    if contain_ok:
        counters["correct_contain"] += 1
    counters["sum_rouge_l"] += rouge_score
    if gt_rationale:
        counters["sum_bert_score"] += bert_score
        counters["n_rationale"] += 1

    # ---- Option-index 匹配 ----
    option_ok = False
    if choices and isinstance(answer_idx, int):
        counters["mc_total"] += 1
        per_dataset[ds_name]["mc"] += 1
        matched_idx = best_option_match(pred_answer, choices)
        if matched_idx is not None and matched_idx == answer_idx:
            counters["correct_option"] += 1
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
        print(f"[{global_idx}/{n_total}] [{ds_name}] EM=FALSE  "
              f"contain={contain_ok}  option={option_ok}  "
              f"ROUGE-L={rouge_score:.3f}  BERTScore={bert_score:.3f}")
        print(f"  Q: {sample['question'][:120]}")
        print(f"  GT answer: {repr(answer_text)}    GT idx: {answer_idx}")
        print(f"  Choices: {choices}")
        print(f"  Pred raw: {pred_text[:200]}")
        print(f"  Pred extracted: {repr(pred_answer)}")


# ============================================================================
# 主评估流程
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估微调后的 VLM 在多测试集上的准确率（批量推理版）"
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

    # ---- 批量推理 ----
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批量推理的 batch size（默认 4，设为 1 则退化为逐条推理）")

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

    # ---- 两阶段参数解析 ----
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

    # 可选：限制每个文件的最大评估数
    if args.max_samples > 0:
        source_counts = defaultdict(int)
        trimmed = []
        for s in samples:
            src = s.get("_source", "")
            if source_counts[src] < args.max_samples:
                trimmed.append(s)
                source_counts[src] += 1
        print(f"--max-samples={args.max_samples}：从 {len(samples)} 条裁剪到 {len(trimmed)} 条\n")
        samples = trimmed

    # ------------------------------------------------------------------
    # 2. 加载模型
    # ------------------------------------------------------------------
    print("加载模型...")
    lora_path = resolve_lora_path(args.lora)
    model, processor = load_model_and_processor(
        args.model, lora_path, local_files_only=True
    )
    print(f"  模型加载完成，batch_size={args.batch_size}\n")

    # ------------------------------------------------------------------
    # 3. 批量评估
    # ------------------------------------------------------------------
    counters = {
        "total": 0,
        "correct_em": 0,
        "correct_contain": 0,
        "correct_option": 0,
        "mc_total": 0,
        "sum_rouge_l": 0.0,
        "sum_bert_score": 0.0,
        "n_rationale": 0,
    }

    per_dataset = defaultdict(lambda: {
        "total": 0, "em": 0, "contain": 0, "option": 0,
        "mc": 0, "rouge_sum": 0.0, "bert_sum": 0.0, "n_rat": 0,
    })

    n_total = len(samples)
    batch_size = args.batch_size
    progress_interval = max(batch_size, 20)

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch = samples[batch_start:batch_end]

        # ---- 3.1 构建 batch 输入 ----
        inputs, _batch_texts, _batch_image_paths, batch_errors = _build_batch_inputs(
            batch, processor, args.test_mode, model.device
        )

        if not inputs:
            continue

        # ---- 3.2 批量生成 ----
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        # ---- 3.3 裁剪并批量解码 ----
        output_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        decoded = processor.batch_decode(
            output_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # ---- 3.4 逐样本评估 ----
        for i, sample in enumerate(batch):
            _evaluate_single_sample(
                pred_text=decoded[i],
                sample=sample,
                error_flag=batch_errors[i],
                args=args,
                counters=counters,
                per_dataset=per_dataset,
                global_idx=batch_start + i + 1,
                n_total=n_total,
            )

        # ---- 3.5 进度打印 ----
        total = counters["total"]
        if total % progress_interval == 0 or batch_end >= n_total:
            bert_avg = counters["sum_bert_score"] / max(counters["n_rationale"], 1)
            print(f"[{total}/{n_total}]  "
                  f"EM={counters['correct_em'] / max(total, 1):.3f}  "
                  f"contain={counters['correct_contain'] / max(total, 1):.3f}  "
                  f"ROUGE-L={counters['sum_rouge_l'] / max(total, 1):.3f}  "
                  f"BERTScore={bert_avg:.3f}  "
                  f"option={counters['correct_option'] / max(counters['mc_total'], 1):.3f}")

    # ------------------------------------------------------------------
    # 4. 最终报告
    # ------------------------------------------------------------------
    total = counters["total"]
    mc_total = counters["mc_total"]
    n_rationale = counters["n_rationale"]
    bert_avg = counters["sum_bert_score"] / max(n_rationale, 1)

    print("\n" + "=" * 60)
    print("评估完成 — 汇总报告")
    print("=" * 60)
    print(f"  总样本数               : {total}")
    print(f"  其中多选题             : {mc_total}")
    print(f"  有推理链的样本数       : {n_rationale}")
    print(f"  EM (精确匹配)          : {counters['correct_em']}/{total} = {counters['correct_em'] / max(total, 1):.4f}")
    print(f"  Contain (词边界包含)   : {counters['correct_contain']}/{total} = {counters['correct_contain'] / max(total, 1):.4f}")
    print(f"  ROUGE-L F1 (答案)      : {counters['sum_rouge_l'] / max(total, 1):.4f}")
    print(f"  BERTScore  (推理链)    : {bert_avg:.4f}  (n={n_rationale})")
    if mc_total:
        print(f"  Option-index accuracy  : {counters['correct_option']}/{mc_total} = {counters['correct_option'] / mc_total:.4f}")

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

    print("\n=== TL;DR ===")
    print(f"EM={counters['correct_em'] / max(total, 1):.4f}  "
          f"contain={counters['correct_contain'] / max(total, 1):.4f}  "
          f"ROUGE-L={counters['sum_rouge_l'] / max(total, 1):.4f}  "
          f"BERTScore={bert_avg:.4f}  "
          f"option={counters['correct_option'] / max(mc_total, 1):.4f}")

    # ---- 可选：保存到 JSON ----
    if args.save_results:
        result_data = {
            "total": total,
            "mc_total": mc_total,
            "n_rationale": n_rationale,
            "em": counters["correct_em"],
            "contain": counters["correct_contain"],
            "option": counters["correct_option"],
            "rouge_l_f1": counters["sum_rouge_l"] / max(total, 1),
            "bertscore_f1": counters["sum_bert_score"] / max(n_rationale, 1),
            "batch_size": args.batch_size,
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
