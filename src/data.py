"""
数据处理工具：样本归一化、JSONL 文件加载。

供 eval.py / eval_batch.py / train.py / gen_human_eval.py 等脚本导入使用。
"""

import json
import os


# ============================================================================
# 样本归一化
# ============================================================================

def normalize_sample(sample: dict) -> dict | None:
    """统一不同数据集的字段格式。

    处理各数据集的字段差异：
    - ScienceQA: answer 是选项索引，answer_text 已在预处理时算好
    - MathVista: answer 直接是答案字符串（无 answer_text 字段），
      choices 可能是字符串 "none"
    - ChartQA / DocVQA: answer_text 就是答案，choices 为空

    返回统一格式的 dict，无有效图片或答案时返回 None。
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

    # ---- answer_text ----
    answer_text = sample.get("answer_text") or ""
    answer_idx = sample.get("answer")

    if not answer_text:
        if isinstance(answer_idx, int) and choices and 0 <= answer_idx < len(choices):
            answer_text = str(choices[answer_idx])
        elif isinstance(answer_idx, str) and answer_idx.strip():
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
        "rationale": sample.get("rationale") or "",
    }


# ============================================================================
# JSONL 批量加载
# ============================================================================

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
