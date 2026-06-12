"""
模型加载工具：4-bit QLoRA 模型 + LoRA adapter 加载，checkpoint 自动选择。

供 eval.py / eval_batch.py / gen_human_eval.py / infer.py 等脚本导入使用。
"""

import glob
import os

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


# ============================================================================
# Checkpoint 自动选择
# ============================================================================

def resolve_lora_path(lora_path: str) -> str:
    """如果 lora_path 是目录且含 checkpoint-* 子目录，自动选最新的。"""
    if os.path.isdir(lora_path):
        ckpts = sorted(glob.glob(os.path.join(lora_path, "checkpoint-*")))
        if ckpts:
            print(f"  自动选择 checkpoint: {ckpts[-1]}")
            return ckpts[-1]
    return lora_path


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_processor(
    model_path: str,
    lora_path: str,
    *,
    local_files_only: bool = False,
    use_double_quant: bool = False,
    device_map: str = "auto",
):
    """加载 4-bit 量化基础模型 + LoRA adapter。

    参数:
        model_path:        基础模型路径（本地或 HuggingFace ID）
        lora_path:         LoRA adapter 路径
        local_files_only:  仅使用本地文件（离线环境）
        use_double_quant:  是否启用双重量化
        device_map:        设备分配策略

    返回:
        (model, processor) 元组
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=use_double_quant,
    )

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=local_files_only
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device_map,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        base_model, lora_path, local_files_only=local_files_only
    )
    model.eval()
    return model, processor
