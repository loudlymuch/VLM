"""
评估微调后的 VLM 模型在多个测试集上的准确率。

支持的指标：
  - EM（Exact Match）：标准化后精确匹配，适用于选择题
  - Contain（词边界包含）：target 作为独立单词出现在预测中
  - Option-index accuracy：仅多选题，预测答案与选项的 fuzzy 匹配
  - ROUGE-L F1：最长公共子序列 F1 分数，适用于开放题（rouge-score 库）
  - BLEU-4：4-gram BLEU 分数（nltk smoothing），衡量生成文本质量

使用示例：
  python scripts/eval.py
  python scripts/eval.py --jsonl data/processed/scienceqa_validation.jsonl data/processed/mathvista_validation.jsonl
  python scripts/eval.py --test-mode no_options -v
"""

import argparse
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# ROUGE-L 和 BLEU-4 指标（通过库函数，无需手写）
from rouge_score import rouge_scorer           # pip install rouge-score
from nltk.translate.bleu_score import (        # pip install nltk
    sentence_bleu,
    SmoothingFunction,
)


# ============================================================================
# 第一部分：文本标准化工具
# ============================================================================

def normalize(text: str) -> str:
    """标准化文本用于比较。

    1. 小写化
    2. 合并多余空白字符
    3. 去除首尾标点（保留负号）
    """
    text = text.strip().lower()
    text = " ".join(text.split())
    text = text.strip(".,;:!?\"'()[]{} \t\n\r")
    return text


def fuzzy_match(a: str, b: str) -> float:
    """SequenceMatcher 相似度，[0, 1]。"""
    return SequenceMatcher(None, a, b).ratio()


# ============================================================================
# 第二部分：答案提取
# ============================================================================

def extract_answer(pred_text: str) -> str:
    """从模型输出中提取 Answer: 后面的部分。

    按优先级尝试多种标记，返回第一个匹配内容。
    """
    if not pred_text:
        return ""

    def _first_sentence(text: str) -> str:
        m = re.search(r"[.。；;！!？?\n]", text)
        if m:
            return text[: m.start()].strip()
        return text.strip()

    # 1: 训练格式标记
    for marker in ["answer:", "Answer:", "ANSWER:"]:
        pos = pred_text.lower().find(marker.lower())
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker) :])

    # 2: 中文标记
    for marker in ["答案是", "答案为", "答案：", "回答："]:
        pos = pred_text.find(marker)
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker) :])

    # 3: 英文自然语言标记
    for marker in ["The answer is", "the answer is"]:
        pos = pred_text.find(marker)
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker) :])

    # 4: 兜底 — 取最后一行非空文本
    lines = [l.strip() for l in pred_text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return pred_text


# ============================================================================
# 第三部分：传统匹配策略
# ============================================================================

def is_exact_match(pred: str, target: str) -> bool:
    """标准化后精确匹配（EM）。"""
    return normalize(pred) == normalize(target)


def is_contained(pred: str, target: str) -> bool:
    """检查 target 是否作为独立单词出现在 pred 中（词边界检查）。

    避免 "cat" 错误匹配 "category"。
    """
    p = normalize(pred)
    t = normalize(target)
    if not t or not p:
        return False

    start = 0
    while True:
        idx = p.find(t, start)
        if idx == -1:
            return False
        end = idx + len(t)
        left_ok = idx == 0 or not p[idx - 1].isalnum()
        right_ok = end == len(p) or not p[end].isalnum()
        if left_ok and right_ok:
            return True
        start = idx + 1


def best_option_match(pred_answer: str, choices: list) -> int | None:
    """找出与预测答案最匹配的选项索引。

    精确匹配 → 立即返回；否则 fuzzy 匹配，阈值 0.5。
    """
    if not choices:
        return None

    best_idx = None
    best_score = 0.0

    for i, choice in enumerate(choices):
        choice_str = str(choice).strip()
        if normalize(pred_answer) == normalize(choice_str):
            return i
        score = fuzzy_match(pred_answer, choice_str)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx if best_score >= 0.5 else None


# ============================================================================
# 第四部分：ROUGE-L 和 BLEU-4（库函数封装）
# ============================================================================

# ROUGE-L scorer（复用实例，避免重复创建）
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# BLEU 平滑函数（method1: 对零计数 +1）
_bleu_smoother = SmoothingFunction().method1


def compute_rouge_l(pred: str, target: str) -> float:
    """计算 ROUGE-L F1 分数（rouge-score 库）。

    ROUGE-L 基于最长公共子序列（LCS），衡量预测和参考答案的文本重叠。
    使用字符级匹配，同时适用于中文和英文。
    """
    norm_pred = normalize(pred)
    norm_target = normalize(target)
    if not norm_pred or not norm_target:
        return 0.0

    scores = _rouge_scorer.score(norm_target, norm_pred)
    return scores["rougeL"].fmeasure  # F1 ∈ [0, 1]


def compute_bleu_4(pred: str, target: str) -> float:
    """计算 BLEU-4 分数（nltk 库，带 smoothing）。

    BLEU 衡量预测文本的 n-gram 精确率，并施加简短惩罚。
    Smoothing 处理短答案中高阶 n-gram 为零的问题。
    """
    pred_tokens = normalize(pred).split()
    ref_tokens = [normalize(target).split()]  # nltk 要求 reference 是嵌套列表

    if not pred_tokens or not ref_tokens[0]:
        return 0.0

    # weights=(0.25,0.25,0.25,0.25) 表示 1~4 gram 等权几何平均
    return sentence_bleu(
        ref_tokens, pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=_bleu_smoother,
    )


# ============================================================================
# 第五部分：数据加载与归一化
# ============================================================================

def normalize_sample(sample: dict) -> dict | None:
    """统一不同数据集的字段格式（与 train.py 的 normalize_sample 逻辑一致）。

    处理各数据集的字段差异：
    - ScienceQA: answer 是选项索引，answer_text 已在预处理时算好
    - MathVista: answer 直接是答案字符串（无 answer_text 字段），
      choices 可能是字符串 "none"
    - ChartQA / DocVQA: answer_text 就是答案，choices 为空
    """
    image_path = sample.get("image")
    if not image_path:
        return None

    question = sample.get("question", "") or ""

    # ---- choices ----
    choices = sample.get("choices") or []
    if isinstance(choices, str):       # MathVista 的 "none"
        choices = []
    if not isinstance(choices, list):
        choices = []

    # ---- answer_text（核心：处理跨数据集差异）----
    answer_text = sample.get("answer_text") or ""
    answer_idx = sample.get("answer")

    if not answer_text:
        if isinstance(answer_idx, int) and choices and 0 <= answer_idx < len(choices):
            # 兜底：整数索引 → 从 choices 取出文字
            answer_text = str(choices[answer_idx])
        elif isinstance(answer_idx, str) and answer_idx.strip():
            # MathVista 主路径：answer 就是答案字符串
            answer_text = answer_idx.strip()
            answer_idx = None
        elif answer_idx is not None:
            answer_text = str(answer_idx).strip()

    if not answer_text:
        return None

    return {
        "image": image_path,
        "question": question,
        "choices": choices,
        "answer": answer_idx if isinstance(answer_idx, int) else None,
        "answer_text": answer_text,
    }


def load_jsonl_files(paths: list[str]) -> tuple[list[dict], dict[str, int]]:
    """加载多个 JSONL 文件，归一化后返回样本列表。

    参数:
        paths: JSONL 文件路径列表

    返回:
        (samples, file_stats)
        samples:     归一化后的样本列表（每个样本含 _source 字段标记来源）
        file_stats:  {文件名: 有效样本数}
    """
    all_samples = []
    file_stats = {}

    for path in paths:
        if not os.path.exists(path):
            print(f"  [WARN] 文件不存在，已跳过: {path}")
            continue

        raw = 0
        kept = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw += 1
                sample = json.loads(line)
                normed = normalize_sample(sample)
                if normed is not None:
                    normed["_source"] = os.path.basename(path)
                    all_samples.append(normed)
                    kept += 1

        file_stats[os.path.basename(path)] = kept
        print(f"  {os.path.basename(path)}: 加载 {raw} 条, 保留 {kept} 条")

    return all_samples, file_stats


# ============================================================================
# 第六部分：Prompt 构建（与 train.py 一致）
# ============================================================================

def build_prompt(question: str, choices: list = None) -> str:
    """根据问题和选项构建测试 prompt。"""
    has_choices = choices is not None and len(choices) > 0

    if has_choices:
        options_text = "Candidate options: " + ", ".join(choices)
        instruction = (
            f"Considering the following options: [{options_text}], "
            "analyze the image and solve the question."
        )
    else:
        instruction = (
            "Analyze the image and provide a direct answer to "
            "the following scientific question."
        )

    return (
        f"{instruction}\n\n"
        f"Question: {question}\n\n"
        "Your output must strictly follow this format:\n"
        "Reasoning: <your step-by-step scientific analysis>\n"
        "Answer: <the specific result or term>"
    )


# ============================================================================
# 第七部分：主评估流程
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

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 7.1 加载数据
    # ------------------------------------------------------------------
    print("=" * 60)
    print("加载测试数据...")
    samples, file_stats = load_jsonl_files(args.jsonl)
    print(f"总计保留 {len(samples)} 条有效样本\n")

    if not samples:
        print("错误：无有效样本，退出。")
        return

    # ------------------------------------------------------------------
    # 7.2 加载模型
    # ------------------------------------------------------------------
    print("加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 如果 --lora 是目录且含 checkpoint-* 子目录，自动选最新的
    lora_path = args.lora
    if os.path.isdir(lora_path):
        import glob as _glob
        ckpts = sorted(_glob.glob(os.path.join(lora_path, "checkpoint-*")))
        if ckpts:
            lora_path = ckpts[-1]
            print(f"  自动选择 checkpoint: {lora_path}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=True)
    model.eval()
    print("  模型加载完成\n")

    # ------------------------------------------------------------------
    # 7.3 逐样本评估
    # ------------------------------------------------------------------
    # 全局计数器
    total = 0
    correct_em = 0
    correct_contain = 0
    correct_option = 0
    mc_total = 0
    sum_rouge_l = 0.0
    sum_bleu_4 = 0.0

    # 按数据集分组
    per_dataset = defaultdict(lambda: {
        "total": 0, "em": 0, "contain": 0, "option": 0,
        "mc": 0, "rouge_sum": 0.0, "bleu_sum": 0.0,
    })

    n_total = len(samples)

    for idx, sample in enumerate(samples):
        image_path = sample["image"]
        question = sample["question"]
        choices = sample["choices"]
        answer_idx = sample["answer"]
        answer_text = sample["answer_text"]
        source = sample.get("_source", "unknown")

        # 数据集简称，如 "scienceqa_validation.jsonl" → "scienceqa_validation"
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
        bleu_score = compute_bleu_4(pred_answer, answer_text)

        # ---- 累计全局 ----
        total += 1
        if em_ok:
            correct_em += 1
        if contain_ok:
            correct_contain += 1
        sum_rouge_l += rouge_score
        sum_bleu_4 += bleu_score

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
        per_dataset[ds_name]["bleu_sum"] += bleu_score

        # ---- 详细日志 ----
        if args.verbose and not em_ok:
            print(f"\n{'=' * 60}")
            print(f"[{total}/{n_total}] [{ds_name}] EM=FALSE  "
                  f"contain={contain_ok}  option={option_ok}  "
                  f"ROUGE-L={rouge_score:.3f}  BLEU-4={bleu_score:.3f}")
            print(f"  Q: {question[:120]}")
            print(f"  GT answer: {repr(answer_text)}    GT idx: {answer_idx}")
            print(f"  Choices: {choices}")
            print(f"  Pred raw: {pred_text[:200]}")
            print(f"  Pred extracted: {repr(pred_answer)}")

        # ---- 进度 ----
        if total % 20 == 0:
            print(f"[{total}/{n_total}]  "
                  f"EM={correct_em / total:.3f}  "
                  f"contain={correct_contain / total:.3f}  "
                  f"ROUGE-L={sum_rouge_l / total:.3f}  "
                  f"BLEU-4={sum_bleu_4 / total:.3f}  "
                  f"option={correct_option / max(mc_total, 1):.3f}")

    # ------------------------------------------------------------------
    # 7.4 最终报告
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("评估完成 — 汇总报告")
    print("=" * 60)
    print(f"  总样本数               : {total}")
    print(f"  其中多选题             : {mc_total}")
    print(f"  EM (精确匹配)          : {correct_em}/{total} = {correct_em / total:.4f}")
    print(f"  Contain (词边界包含)   : {correct_contain}/{total} = {correct_contain / total:.4f}")
    print(f"  ROUGE-L F1 (均值)      : {sum_rouge_l / total:.4f}")
    print(f"  BLEU-4    (均值)       : {sum_bleu_4 / total:.4f}")
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
            print(f"    BLEU-4      : {st['bleu_sum'] / n:.4f}")
            if st["mc"]:
                print(f"    Option-Index: {st['option']}/{st['mc']} = {st['option'] / st['mc']:.4f}")

    # ---- 一行摘要 ----
    print("\n=== TL;DR ===")
    print(f"EM={correct_em / total:.4f}  "
          f"contain={correct_contain / total:.4f}  "
          f"ROUGE-L={sum_rouge_l / total:.4f}  "
          f"BLEU-4={sum_bleu_4 / total:.4f}  "
          f"option={correct_option / max(mc_total, 1):.4f}")

    # ---- 可选：保存到 JSON ----
    if args.save_results:
        result_data = {
            "total": total,
            "mc_total": mc_total,
            "em": correct_em,
            "contain": correct_contain,
            "option": correct_option,
            "rouge_l_f1": sum_rouge_l / total,
            "bleu_4": sum_bleu_4 / total,
            "per_dataset": {
                ds: {
                    "total": st["total"],
                    "em": st["em"] / max(st["total"], 1),
                    "contain": st["contain"] / max(st["total"], 1),
                    "rouge_l_f1": st["rouge_sum"] / max(st["total"], 1),
                    "bleu_4": st["bleu_sum"] / max(st["total"], 1),
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
