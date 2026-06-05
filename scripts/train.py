"""
使用 QLoRA 在多个 VQA 数据集上微调 Qwen2.5-VL 视觉语言模型。

本脚本的核心流程：
1. 加载多个 JSONL 数据集文件（ScienceQA、MathVista、ChartQA、DocVQA）
2. 统一不同数据集的字段格式（归一化处理）
3. 使用 4-bit 量化（QLoRA）加载模型，大幅降低显存占用
4. 在模型的关键层上添加 LoRA 适配器（低秩矩阵），只训练这少量参数
5. 使用 Hugging Face Trainer 进行多轮训练
6. 保存训练好的 LoRA 权重（adapter）

运行方式：
    python scripts/train.py                                    # 使用默认参数训练全部数据集
    python scripts/train.py --resume                           # 从最近的 checkpoint 继续训练
    python scripts/train.py --train-data data/processed/scienceqa_train.jsonl  # 只训练 ScienceQA
    python scripts/train.py --epochs 5 --lr 1e-4               # 自定义训练参数
"""

import argparse   # 解析命令行参数，让我们可以在终端灵活配置训练参数
import glob       # 用通配符查找文件，比如查找已有的 checkpoint 目录
import os         # 文件和目录操作
import random     # 控制随机数，用于混合训练中随机隐藏选项

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
# Dataset: 创建一个 HuggingFace 数据集对象（类似表格）
# concatenate_datasets: 把多个数据集拼接成一个
# load_dataset: 从文件加载数据集（支持 JSONL、JSON、CSV 等格式）

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# peft（Parameter-Efficient Fine-Tuning）：高效微调库
# LoraConfig: 配置 LoRA 的超参数（秩 r、缩放因子 alpha、目标层等）
# get_peft_model: 把 LoRA 配置应用到模型上，返回可训练的模型
# prepare_model_for_kbit_training: 准备量化模型用于训练（处理权重冻结等）

from transformers import (
    AutoProcessor,                        # 自动加载与模型匹配的处理器（tokenizer + 图像处理器）
    BitsAndBytesConfig,                   # 4-bit 量化的配置（NF4、双重量化等）
    Qwen2_5_VLForConditionalGeneration,   # Qwen2.5-VL 模型本身（因果语言模型）
    TrainingArguments,                    # 训练超参数（学习率、batch size、epochs 等）
    Trainer,                              # HuggingFace 的训练器，封装了训练循环
)
from qwen_vl_utils import process_vision_info
# Qwen 官方工具：从对话消息中提取图像和视频信息


# ============================================================================
# 第一部分：构建 Prompt（提示词）
# ============================================================================

def build_prompt(question: str, choices: list = None) -> str:
    """
    根据问题和选项构建发送给模型的提示词（prompt）。

    参数:
        question: 问题文本，例如 "What is the capital of France?"
        choices:  候选选项列表，例如 ["London", "Paris", "Berlin"]
                 如果是 None 或空列表，表示这是一道开放题，没有选项

    返回:
        完整的 prompt 字符串，模型将根据这个 prompt 生成答案

    提示词结构:
        [根据有无选项选择合适的指令]
        Question: <问题>
        Your output must strictly follow this format:
        Reasoning: <你的逐步推理过程>
        Answer: <最终答案>
    """
    # 检查是否有有效选项（非 None 且非空列表）
    has_choices = choices is not None and len(choices) > 0

    if has_choices:
        # ---- 有选项的情况：把候选答案列出来 ----
        # 用逗号分隔所有选项，例如 "London, Paris, Berlin"
        options_text = "Candidate options: " + ", ".join(choices)
        instruction = (
            f"Considering the following options: [{options_text}], "
            "analyze the image and solve the question."
        )
    else:
        # ---- 没有选项的情况（开放题）：让模型自由回答 ----
        instruction = "Analyze the image and provide a direct answer to the following scientific question."

    # 返回完整的 prompt，统一要求 Reason + Answer 格式
    return (
        f"{instruction}\n\n"
        f"Question: {question}\n\n"
        "Your output must strictly follow this format:\n"
        "Reasoning: <your step-by-step scientific analysis>\n"
        "Answer: <the specific result or term>"
    )


# ============================================================================
# 第二部分：数据归一化（统一不同数据集的字段格式）
# ============================================================================

def normalize_sample(sample: dict) -> dict | None:
    """
    把不同数据集的样本统一为相同的字段格式。

    背景：四个数据集的字段名和含义不完全一样——
    - ScienceQA:  answer 字段是选项索引（如 0/1/2/3），
                  但 answer_text 已在预处理时从 choices[answer] 解析好，直接可用
    - MathVista:  answer 直接就是答案字符串，没有 answer_text 字段
                  choices 可能是空字符串 "none"（需转为空列表）
    - ChartQA / DocVQA: answer_text 就是答案，choices 为空列表，无推理链

    本函数的任务就是把这些差异"抹平"，
    让下游的 Collator 和 Trainer 看到的都是统一格式。

    参数:
        sample: 原始样本字典，字段因数据集而异

    返回:
        归一化后的字典，字段固定为: image, question, choices, answer, answer_text, rationale
        如果样本缺少图片或答案，返回 None（该样本被丢弃）
    """
    # ---- 1. 提取图片路径 ----
    # 所有样本必须有图片，没有图片的样本直接丢弃
    image_path = sample.get("image")
    if not image_path:
        return None

    # ---- 2. 提取问题 ----
    question = sample.get("question", "") or ""

    # ---- 3. 处理选项（choices）----
    choices = sample.get("choices") or []
    # MathVista 在无选项时 choices 是字符串 "none"，需要转成空列表
    if isinstance(choices, str):
        choices = []
    # 兜底：确保 choices 一定是列表类型
    if not isinstance(choices, list):
        choices = []

    # ---- 4. 提取答案文本（这是最重要的部分）----
    answer_text = sample.get("answer_text") or ""   # 先尝试获取 answer_text 字段
    answer_idx = sample.get("answer")                # 再获取 answer 字段（可能是索引或字符串）

    # 如果 answer_text 为空，尝试从 answer 字段推断
    # 注意：正常处理的 ScienceQA / ChartQA / DocVQA 的 answer_text 已在预处理时填好，
    # 不会进入此分支。这里只处理"answer_text 意外缺失"的兜底情况
    if not answer_text:
        if isinstance(answer_idx, int) and choices and 0 <= answer_idx < len(choices):
            # 情况 A：answer 是整数索引 → 从 choices 列表中取出对应文字
            # 例如 answer=2, choices=["cat","dog","bird","fish"] → answer_text="bird"
            answer_text = str(choices[answer_idx])

        elif isinstance(answer_idx, str) and answer_idx.strip():
            # 情况 B：answer 直接就是答案字符串（MathVista 主路径在此）
            # 例如 answer="42" → answer_text="42"
            answer_text = answer_idx.strip()
            answer_idx = None   # 转为开放题格式

        elif answer_idx is not None:
            # 情况 C（兜底）：answer 是其他类型（如 float），直接转字符串
            answer_text = str(answer_idx).strip()

    # 如果经过以上处理仍然没有答案，丢弃该样本
    if not answer_text:
        return None

    # ---- 5. 提取推理过程（rationale）----
    # ScienceQA 有 lecture+solution 作为推理链，其他数据集通常为空
    rationale = sample.get("rationale") or ""

    # ---- 6. 返回统一格式的样本 ----
    return {
        "image": image_path,
        "question": question,
        "choices": choices,                                                    # 始终是列表（可能为空）
        "answer": answer_idx if isinstance(answer_idx, int) else None,         # 仅选择题有整数索引
        "answer_text": answer_text,                                            # 始终是字符串
        "rationale": rationale,                                                # 推理链（可能为空）
    }


# ============================================================================
# 第三部分：批量加载和拼接多个 JSONL 数据集
# ============================================================================

def load_datasets(paths: list[str]) -> Dataset:
    """
    加载一个或多个 JSONL 文件，归一化后拼接成一个大数据集。

    参数:
        paths: JSONL 文件路径列表，例如:
               ["data/processed/scienceqa_train.jsonl",
                "data/processed/mathvista_train.jsonl"]

    返回:
        拼接后的 HuggingFace Dataset 对象，所有样本字段已统一
    """
    all_parts = []          # 存放每个数据集（归一化后）的列表
    total_loaded = 0        # 统计总共读取了多少条原始数据
    total_kept = 0          # 统计归一化后保留了多少条

    for path in paths:
        # 跳过不存在的文件（比如还没下载或处理的数据集）
        if not os.path.exists(path):
            print(f"  [WARN] 文件不存在，已跳过: {path}")
            continue

        # HuggingFace datasets 库的 load_dataset 可以直接读取 JSONL 文件
        # "json" 是数据格式，"data_files" 指定文件路径，split="data" 表示加载全部数据
        ds = load_dataset("json", data_files={"data": path}, split="data")
        raw_count = len(ds)
        total_loaded += raw_count

        # 对每条数据进行归一化，过滤掉无效样本
        kept_samples = []
        for sample in ds:
            normed = normalize_sample(sample)
            if normed is not None:
                kept_samples.append(normed)

        kept = len(kept_samples)
        total_kept += kept

        if kept == 0:
            print(f"  {path}: 加载 {raw_count} 条, 保留 0 条（全部被过滤）")
            continue

        # 将归一化后的字典列表转为 HuggingFace Dataset 对象
        # Dataset.from_dict 接受 {"列名1": [值列表], "列名2": [值列表]} 的格式
        part_ds = Dataset.from_dict(
            {k: [s[k] for s in kept_samples] for k in kept_samples[0].keys()}
        )
        print(f"  {path}: 加载 {raw_count} 条, 保留 {kept} 条（过滤 {raw_count - kept} 条）")
        all_parts.append(part_ds)

    # 安全检查：如果所有文件都不存在或数据全部被过滤，报错退出
    if not all_parts:
        raise RuntimeError("没有找到有效的训练数据！请先运行数据处理脚本（process_*.py）。")

    # 将所有数据集拼接成一个大的数据集
    # 拼接要求所有数据集的列名相同 —— 我们的 normalize_sample 保证了这一点
    full_ds = concatenate_datasets(all_parts)
    print(f"总计: 加载 {total_loaded} 条, 保留 {total_kept} 条, 最终数据集大小 {len(full_ds)} 条")
    return full_ds


# ============================================================================
# 第四部分：评估指标（可选，目前未启用）
# ============================================================================

def compute_metrics(eval_pred):
    """
    计算 token 级别的准确率。

    这个函数目前没有被 Trainer 调用（eval 被注释掉了），
    但保留作为后续可能需要评估时的参考。

    参数:
        eval_pred: (predictions, labels) 元组
            predictions: 模型输出的 logits（未归一化的概率）
            labels: 真实的 token ID 序列（-100 表示被掩码的位置）

    返回:
        {"token_acc": 0.75} 形式的字典
    """
    predictions, labels = eval_pred
    # predictions 可能是 tuple（包含多个输出），取第一个
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    # argmax 取每个位置概率最大的 token ID
    pred_ids = np.argmax(predictions, axis=-1)
    # 只计算非掩码位置的准确率（labels == -100 的位置不参与计算）
    mask = labels != -100
    if mask.sum() == 0:
        return {"token_acc": 0.0}
    correct = (pred_ids == labels) & mask
    token_acc = correct.sum() / mask.sum()
    return {"token_acc": float(token_acc)}


# ============================================================================
# 第五部分：数据整理器（Collator）—— 训练中最关键的自定义组件
# ============================================================================

class VisionDataCollator:
    """
    视觉语言模型的数据整理器。

    作用：把一个 batch 中的多个样本拼接成模型可以一次性处理的形式。

    主要工作：
    1. 为每个样本构建 prompt 和完整对话（prompt + 答案）
    2. 应用 Qwen 的 chat template（对话模板）将对话转为文本
    3. 处理图像（读取、resize、编码为视觉 token）
    4. 计算 prompt 的长度，用于后续掩码（让模型只在答案部分学习）
    5. 将 batch 内所有样本 padding 到相同长度

    为什么需要自定义 Collator？
    - 默认的 Collator 不知道如何处理图像
    - 我们需要精确控制 labels 中哪些位置被掩码（不计算 loss）
      —— 只对 "答案部分" 计算 loss，"问题部分" 不参与
    """

    def __init__(self, processor, nochoice_random=0.2):
        """
        参数:
            processor: 模型的处理器（tokenizer + 图像处理器）
            nochoice_random: 混合训练中隐藏选项的概率（0.2 = 20% 的样本不提供选项）
        """
        self.processor = processor
        self.nochoice_random = nochoice_random

    def __call__(self, batch):
        """
        每个训练 step 都会调用这个方法。

        参数:
            batch: 一个列表，包含 batch_size 个样本字典

        返回:
            一个字典 {"input_ids": ..., "labels": ..., "attention_mask": ...}
            可以直接喂给模型
        """
        texts = []       # 每个样本的完整文本（prompt + 答案）
        images = []      # 每个样本的图像（已处理为模型需要的格式）
        prompt_lens = [] # 每个样本的 prompt 长度（token 数），用于掩码

        for sample in batch:
            # ---- 获取样本的基本字段 ----
            image_path = sample.get("image")
            if not image_path:
                continue

            question = sample.get("question", "")
            choices = sample.get("choices", [])
            answer_text = sample.get("answer_text") or ""
            if not answer_text:
                continue

            # ---- 混合训练（Mixed Training）逻辑 ----
            # 以 nochoice_random 的概率随机把选项藏起来
            # 这样做的好处：模型既学会了"看选项选答案"，
            # 也学会了"没有选项时自己推理"，测试时两种模式都能应对
            current_choices = choices
            if random.random() < self.nochoice_random:
                current_choices = None   # 隐藏选项，当作开放题训练

            # ---- 构建 prompt 和回答 ----
            prompt = build_prompt(question, current_choices)

            rationale = sample.get("rationale") or ""
            if rationale:
                # 有推理链的数据集（如 ScienceQA）：Reasoning + Answer
                final_response = f"Reasoning: {rationale}\nAnswer: {answer_text}"
            else:
                # 没有推理链的数据集（如 ChartQA）：只有 Answer
                final_response = f"Answer: {answer_text}"

            # ---- 构建对话格式 ----
            # messages_prompt: 只有用户消息（用于计算 prompt 长度）
            # messages_full: 用户消息 + 模型回答（用于完整的训练序列）
            messages_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},  # 图片
                        {"type": "text", "text": prompt},       # 文字 prompt
                    ],
                },
            ]
            messages_full = messages_prompt + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_response}],
                }
            ]

            # ---- 应用对话模板（chat template）----
            # 把对话消息列表转换为模型训练时使用的文本格式
            # add_generation_prompt=True 会在末尾添加生成提示（如 "<|assistant|>"）
            prompt_text = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )

            # ---- 处理视觉信息 ----
            # process_vision_info 从消息中提取图像并转换为模型需要的 tensor 格式
            prompt_image_inputs, _ = process_vision_info(messages_prompt)
            image_inputs, _ = process_vision_info(messages_full)
            if not image_inputs:
                continue

            # ---- 计算 prompt 部分的 token 长度 ----
            # 这是为了后续把 prompt 的 labels 设为 -100（不计算 loss）
            # 原理：先单独把 prompt 文本 tokenize，得到它的 token 数
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=prompt_image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            prompt_len = prompt_inputs.input_ids.size(1)  # prompt 的 token 数量

            texts.append(full_text)
            images.append(image_inputs[0])   # process_vision_info 返回列表，取第一个
            prompt_lens.append(prompt_len)

        # 如果整个 batch 都没有有效样本，返回空字典
        if not texts:
            return {}

        # ---- 批量 tokenize ----
        # 将整个 batch 的文本和图像一起处理
        inputs = self.processor(
            text=texts,
            images=images,
            videos=None,
            padding=True,            # 将不同长度的序列 padding 到相同长度
            return_tensors="pt",     # 返回 PyTorch tensor
        )

        # ---- 构建 labels（训练目标）----
        # labels 初始就是 input_ids 的拷贝
        input_ids = inputs.input_ids
        labels = input_ids.clone()

        # 第一步：将 padding token 对应的 label 设为 -100
        # -100 是 PyTorch 交叉熵损失的默认忽略索引，模型不会在这些位置学习
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # 第二步：将 prompt 部分的 label 也设为 -100
        # 这样模型只在 "答案部分" 学习，不会在 "问题部分" 学习
        # 这很重要：我们不希望模型去"背诵"问题，只希望它学会"回答"
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs


# ============================================================================
# 第六部分：主函数 —— 组装一切，启动训练
# ============================================================================

def main() -> None:
    """
    训练的主入口。

    流程概览:
    1. 解析命令行参数
    2. 加载并归一化多个数据集
    3. 以 4-bit 量化加载基础模型
    4. 在模型上添加 LoRA 适配器
    5. 配置 Trainer 并开始训练
    6. 保存训练好的 LoRA 权重
    """

    # ------------------------------------------------------------------
    # 6.1 命令行参数
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="QLoRA 多数据集微调")

    # 模型和数据路径
    parser.add_argument("--model", default="/mnt/workspace/vlm-model-dir",
                        help="基础模型的路径（本地目录或 HuggingFace 模型名）")
    parser.add_argument("--train-data", nargs="+",  # nargs="+" 表示接受 1 个或多个值
                        default=[
                            "data/processed/scienceqa_train.jsonl",
                            "data/processed/mathvista_train.jsonl",
                            "data/processed/chartqa_train.jsonl",
                            "data/processed/docvqa_train.jsonl",
                        ],
                        help="训练用的 JSONL 文件列表")
    parser.add_argument("--eval-data", nargs="+", default=None,
                        help="验证用的 JSONL 文件列表（可选）")
    parser.add_argument("--output-dir", default="outputs/checkpoints/qlora_multitask",
                        help="模型 checkpoint 和最终权重的保存目录")

    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每个 GPU 每步处理的样本数")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="梯度累积步数（有效 batch = batch_size × grad_accum）")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率（2e-4 = 0.0002）")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数（整个数据集被遍历的次数）")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="最大训练步数（-1 表示使用 epochs 控制）")
    parser.add_argument("--resume", action="store_true",
                        help="从最近的 checkpoint 继续训练")

    # 混合训练和数据处理
    parser.add_argument("--nochoice-random", type=float, default=0.2,
                        help="训练时随机隐藏选项的概率（0=始终显示，1=始终隐藏）")
    parser.add_argument("--save-steps", type=int, default=300,
                        help="每隔多少步保存一次 checkpoint")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="每隔多少步打印一次 loss 日志")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证实验可复现")
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28,
                        help="图像最大像素数（1024×28×28 = 802816），超出会被缩放")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28,
                        help="图像最小像素数（256×28×28 = 200704），不足会被放大")

    args = parser.parse_args()

    # ---- 设置随机种子，确保训练可复现 ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 6.2 加载数据集
    # ------------------------------------------------------------------
    print("正在加载训练数据...")
    train_dataset = load_datasets(args.train_data)

    eval_dataset = None
    if args.eval_data:
        print("正在加载验证数据...")
        eval_dataset = load_datasets(args.eval_data)

    # ------------------------------------------------------------------
    # 6.3 配置 4-bit 量化并加载模型
    # ------------------------------------------------------------------
    # BitsAndBytesConfig 是 QLoRA 的核心：
    # - load_in_4bit=True: 把模型权重从 16-bit float 压缩到 4-bit 整数
    #   原本 7B 模型需要 ≈14GB 显存（bf16），4-bit 后只需 ≈4GB
    # - bnb_4bit_quant_type="nf4": 使用 NF4（Normal Float 4）量化格式
    #   这是专门为神经网络权重设计的，信息损失比普通 4-bit 更小
    # - bnb_4bit_compute_dtype=torch.bfloat16:
    #   计算时临时反量化到 bfloat16，计算完再量化为 4-bit
    #   这样既省显存，又保证了计算精度
    # - bnb_4bit_use_double_quant=True:
    #   双重量化：对量化参数本身再做一次量化
    #   每个参数额外节省约 0.4 bit，对精度影响极小
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # AutoProcessor: 根据模型名称自动加载对应的处理器
    # 处理器包含两部分：
    # 1. tokenizer: 将文本转为 token ID 序列
    # 2. image_processor: 将图像 resize、归一化后转为模型可接受的 tensor
    # min_pixels / max_pixels 控制图像缩放范围，平衡显存和图像细节
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    # 加载基础模型
    # - torch_dtype=torch.bfloat16: 计算精度（只在反量化时使用）
    # - device_map="auto": 自动将模型各层分配到可用的 GPU（支持单卡/多卡）
    # - low_cpu_mem_usage=True: 加载时尽量少用 CPU 内存
    # - quantization_config: 应用上面的 4-bit 量化配置
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    # 准备量化模型用于训练
    # 主要工作：冻结量化层、转换某些不支持梯度计算的层
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 6.4 配置 LoRA（Low-Rank Adaptation）
    # ------------------------------------------------------------------
    # LoRA 的核心思想：
    # 不在原始的 7B 参数上训练，而是插入"低秩矩阵"（小矩阵），
    # 只训练这些小矩阵。原始权重保持冻结（不更新）。
    #
    # 可训练参数量: 约 0.01% ~ 0.1%，从 7B 降到 10M 左右
    #
    # 参数说明：
    # - r=16: LoRA 的秩（rank），控制低秩矩阵的维度
    #   值越大 → 表达能力越强，但训练参数越多
    #   常用范围：8/16/32，16 是性价比最均衡的选择
    # - lora_alpha=32: 缩放因子，实际学习率被缩放为 alpha/r = 32/16 = 2
    #   值越大 → LoRA 对原始输出的"影响力"越大
    # - lora_dropout=0.05: LoRA 层中的 dropout 比例，防止过拟合
    #   5% 的神经元会在训练时随机"关闭"
    # - target_modules: 要添加 LoRA 适配器的目标层
    #   q_proj: 查询投影（注意力机制中的 Q）
    #   k_proj: 键投影（注意力机制中的 K）
    #   v_proj: 值投影（注意力机制中的 V）
    #   o_proj: 输出投影（注意力机制后的线性层）
    #   gate_proj/up_proj/down_proj: FFN（前馈网络）中的门控和投影层
    #   覆盖全部 7 个关键投影层，最大化微调效果
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",          # 不额外训练 bias 参数
        task_type="CAUSAL_LM", # 任务类型：因果语言模型（自回归生成）
    )

    # 将 LoRA 配置注入模型
    # 这时模型的结构变为：原始冻结权重 + 可训练的 LoRA 低秩矩阵
    model = get_peft_model(model, lora_config)

    # 打印可训练参数量（方便检查 LoRA 是否生效）
    model.print_trainable_parameters()

    # 切换到训练模式（开启 dropout 等）
    model.train()

    # ------------------------------------------------------------------
    # 6.5 创建数据整理器和训练参数
    # ------------------------------------------------------------------
    data_collator = VisionDataCollator(processor, nochoice_random=args.nochoice_random)

    # TrainingArguments 集中管理所有训练相关的超参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,                  # checkpoint 保存路径
        per_device_train_batch_size=args.batch_size, # 每 GPU 的 batch size
        gradient_accumulation_steps=args.grad_accum, # 梯度累积步数
        # 有效 batch size = batch_size × grad_accum = 1 × 8 = 8
        # 梯度累积的意思是：每 grad_accum 步才更新一次参数
        # 这样即使用小 batch size 也能获得大 batch 的效果
        learning_rate=args.lr,                       # 学习率
        max_steps=args.max_steps,                    # 最大步数（-1 = 用 epochs）
        logging_steps=args.logging_steps,            # 每 N 步打一次 log
        save_steps=args.save_steps,                  # 每 N 步存一次 checkpoint
        fp16=False,                                  # 不使用 fp16（混合精度用 bf16）
        bf16=True,                                   # 使用 bfloat16 计算（数值范围大，不易溢出）
        remove_unused_columns=False,                 # 不删除数据集中未用的列（我们需要 image 等字段）
        report_to=[],                                # 不上报训练日志到 wandb/tensorboard
        lr_scheduler_type="cosine",                  # 余弦退火学习率调度
        # 学习率会从初始值按余弦曲线逐渐降到接近 0
        warmup_ratio=0.1,                            # 前 10% 的步数线性增加学习率（warmup）
        # warmup 可以避免训练初期因为学习率太大导致的不稳定
        gradient_checkpointing=True,                 # 梯度检查点：用计算换显存
        # 不保存所有中间激活值，反向传播时重新计算，节省约 30% 显存
        optim="paged_adamw_8bit",                   # 使用 8-bit AdamW 优化器
        # 优化器状态也被量化到 8-bit，进一步节省显存
        save_total_limit=3,                         # 只保留最近 3 个 checkpoint，旧的自动删除
        save_strategy="steps",                      # 按步数（而非 epoch）保存
        num_train_epochs=args.epochs,               # 训练轮数
        seed=args.seed,                              # 随机种子
    )

    # ------------------------------------------------------------------
    # 6.6 创建 Trainer 并开始训练
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,                         # 带 LoRA 的模型
        args=training_args,                  # 训练超参数
        train_dataset=train_dataset,         # 训练集
        eval_dataset=eval_dataset,           # 验证集（None 表示不评估）
        data_collator=data_collator,         # 自定义数据整理器
        processing_class=processor.tokenizer, # tokenizer（用于保存时的配置）
    )

    # 判断是否需要从 checkpoint 恢复训练
    # --resume 参数 或 输出目录中已存在 checkpoint-* 子目录 都会触发续训
    resume = args.resume or (
        len(glob.glob(os.path.join(args.output_dir, "checkpoint-*"))) > 0
    )

    # 开始训练！
    # resume_from_checkpoint=True: 从最新的 checkpoint 加载优化器状态、学习率等
    # 这会跳过已完成的步数，从上次中断的地方继续
    trainer.train(resume_from_checkpoint=resume)

    # ------------------------------------------------------------------
    # 6.7 保存最终模型
    # ------------------------------------------------------------------
    # 保存 LoRA 权重（adapter_config.json + adapter_model.safetensors）
    # 注意：只保存 LoRA 的"增量权重"，不保存整个 7B 模型
    # 保存后的文件大小通常只有几十 MB
    model.save_pretrained(args.output_dir)

    # 保存 processor（tokenizer 配置 + 图像处理器配置）
    # 推理时需要加载同样的 processor 来处理输入
    processor.save_pretrained(args.output_dir)

    print(f"训练完成！LoRA adapter 已保存到: {args.output_dir}")


# ============================================================================
# 脚本入口
# ============================================================================

if __name__ == "__main__":
    main()
