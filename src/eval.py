"""
评估工具函数：文本标准化、答案提取、匹配策略、推理链提取、指标计算、Prompt 构建。

供 eval.py / eval_batch.py / gen_human_eval.py 等脚本导入使用。
"""

import re
from difflib import SequenceMatcher

import torch
from rouge_score import rouge_scorer   # pip install rouge-score
from bert_score import BERTScorer     # pip install bert-score


# ============================================================================
# 模块级 scorer 实例（复用，避免每次调用重新创建）
# ============================================================================

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

_bert_scorer = BERTScorer(
    model_type="roberta-large",
    lang="en",
    device="cuda" if torch.cuda.is_available() else "cpu",
    rescale_with_baseline=True,
)


# ============================================================================
# 第一部分：文本标准化工具
# ============================================================================

def normalize(text: str) -> str:
    """标准化文本用于比较：小写化、合并空白、去首尾标点（保留负号）。"""
    text = text.strip().lower()
    text = " ".join(text.split())
    text = text.strip(".,;:!?\"'()[]{} \t\n\r")
    return text


def fuzzy_match(a: str, b: str) -> float:
    """SequenceMatcher 相似度，[0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


# ============================================================================
# 第二部分：答案提取
# ============================================================================

def _first_sentence(text: str) -> str:
    """截取到第一个句尾标点之前的内容。"""
    m = re.search(r"[.。；;！!？?\n]", text)
    if m:
        return text[: m.start()].strip()
    return text.strip()


def extract_answer(pred_text: str) -> str:
    """从模型输出中提取 Answer: 后面的部分。

    按优先级尝试多种标记，返回第一个匹配内容。
    """
    if not pred_text:
        return ""

    # 1: 训练格式标记
    for marker in ["answer:", "Answer:", "ANSWER:"]:
        pos = pred_text.lower().find(marker.lower())
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker):])

    # 2: 中文标记
    for marker in ["答案是", "答案为", "答案：", "回答："]:
        pos = pred_text.find(marker)
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker):])

    # 3: 英文自然语言标记
    for marker in ["The answer is", "the answer is"]:
        pos = pred_text.find(marker)
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker):])

    # 4: 兜底 — 取最后一行非空文本
    lines = [l.strip() for l in pred_text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return pred_text


# ============================================================================
# 第三部分：匹配策略
# ============================================================================

def is_exact_match(pred: str, target: str) -> bool:
    """标准化后精确匹配（EM）."""
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
# 第四部分：推理链提取
# ============================================================================

def extract_reasoning(pred_text: str) -> str:
    """从模型输出中提取 Reasoning: 部分（Answer: 之前的内容）."""
    if not pred_text:
        return ""

    for marker in ["answer:", "Answer:", "ANSWER:"]:
        pos = pred_text.lower().find(marker.lower())
        if pos != -1:
            prefix = pred_text[:pos]
            break
    else:
        return pred_text

    for marker in ["reasoning:", "Reasoning:", "REASONING:"]:
        rpos = prefix.lower().find(marker.lower())
        if rpos != -1:
            return prefix[rpos + len(marker):].strip()

    return prefix.strip()


# ============================================================================
# 第五部分：评估指标（库函数封装）
# ============================================================================

def compute_rouge_l(pred: str, target: str) -> float:
    """ROUGE-L F1 — 基于最长公共子序列，适用于答案文本重叠度评估。"""
    norm_pred = normalize(pred)
    norm_target = normalize(target)
    if not norm_pred or not norm_target:
        return 0.0
    scores = _rouge_scorer.score(norm_target, norm_pred)
    return scores["rougeL"].fmeasure  # F1 ∈ [0, 1]


def compute_bertscore(pred: str, target: str) -> float:
    """BERTScore F1 — 语义相似度，适用于推理链质量评估。"""
    if not pred or not target:
        return 0.0
    P, R, F1 = _bert_scorer.score([pred], [target])
    return float(F1[0])


# ============================================================================
# 第六部分：Prompt 构建
# ============================================================================

def build_prompt(question: str, choices: list = None) -> str:
    """根据问题和选项构建测试 prompt。

    根据有无候选选项选择不同的指令，末尾统一要求 Reasoning + Answer 格式。
    """
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
