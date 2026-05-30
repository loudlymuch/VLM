"""Train the model with a minimal QLoRA setup using Hugging Face Trainer."""
import os
import random
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from qwen_vl_utils import process_vision_info


def build_prompt(question: str, choices: list = None) -> str:
    # 随机决定是否隐藏选项（混合训练逻辑）
    # 这里的 choices 已经在 Collator 里决定了是 None 还是 列表
    has_choices = choices is not None and len(choices) > 0
    
    if has_choices:
        # 不使用 (A)(B)，直接用分号或换行列出
        # 这种方式告诉模型：答案就在这些词里，但你需要自己识别
        options_text = "Candidate options: " + ", ".join(choices)
        instruction = f"Considering the following options: [{options_text}], analyze the image and solve the question."
    else:
        instruction = "Analyze the image and provide a direct answer to the following scientific question."

    return (
        f"{instruction}\n\n"
        f"Question: {question}\n\n"
        "Your output must strictly follow this format:\n"
        "Reasoning: <your step-by-step scientific analysis>\n"
        "Answer: <the specific result or term>"
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    pred_ids = np.argmax(predictions, axis=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return {"token_acc": 0.0}
    correct = (pred_ids == labels) & mask
    token_acc = correct.sum() / mask.sum()
    return {"token_acc": float(token_acc)}


class VisionDataCollator:
    def __init__(self, processor, nochoice_random = 0.2):
        self.processor = processor
        self.nochoice_random = 0.2

    def __call__(self, batch):
            texts = []
            images = []
            prompt_lens = []

            for sample in batch:
                image_path = sample.get("image")
                if not image_path:
                    continue

                question = sample.get("question", "")
                choices = sample.get("choices", [])
                answer_idx = sample.get("answer")
                answer_text = sample.get("answer_text") or ""
                
                # --- 必须保留：确保答案是具体的文本而不是空的 ---
                if not answer_text and isinstance(answer_idx, int) and choices:
                    if 0 <= answer_idx < len(choices):
                        answer_text = str(choices[answer_idx])
                
                if not answer_text: # 如果真的没答案，这条数据就不能训练
                    continue

                # --- 混合训练逻辑 ---
                current_choices = choices
                if random.random() < self.nochoice_random:
                    current_choices = None

                prompt = build_prompt(question, current_choices)

                rationale = sample.get("rationale") or ""
                if rationale:
                    # 统一输出格式
                    final_response = f"Reasoning: {rationale}\nAnswer: {answer_text}"
                else:
                    final_response = f"Answer: {answer_text}"

                # --- 构造对话格式 ---
                messages_prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                messages_full = messages_prompt + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": final_response}],
                    }
                ]

                # 转换文本
                prompt_text = self.processor.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True
                )
                full_text = self.processor.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )

                # 处理视觉信息
                prompt_image_inputs, _ = process_vision_info(messages_prompt)
                image_inputs, _ = process_vision_info(messages_full)
                if not image_inputs:
                    continue

                # 计算 Prompt 长度（为了后续把 Prompt 部分的 Label 设为 -100）
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=prompt_image_inputs,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                prompt_len = prompt_inputs.input_ids.size(1)

                texts.append(full_text)
                images.append(image_inputs[0])
                prompt_lens.append(prompt_len)

            if not texts: # 安全检查
                return {}

            # 批量处理
            inputs = self.processor(
                text=texts,
                images=images,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            
            input_ids = inputs.input_ids
            labels = input_ids.clone()
            
            # 掩码处理：把 Prompt 部分和 Padding 部分的 Label 设为 -100，不计算 Loss
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            
            for i, prompt_len in enumerate(prompt_lens):
                # 将每个样本的 Prompt 部分掩码掉
                labels[i, :prompt_len] = -100
                
            inputs["labels"] = labels
            return inputs

def main() -> None:
    # Simple configuration for beginners
    # model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_name = "/mnt/workspace/vlm-model-dir"
    train_jsonl = "data/processed/scienceqa_train.jsonl"
    eval_jsonl = "data/processed/scienceqa_validation.jsonl"
    output_dir = "outputs/checkpoints/qlora_scienceqa"
    # max_steps = 200
    batch_size = 1
    grad_accum = 8
    lr = 2e-4

    os.makedirs(output_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28  # 限制图片大小
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    dataset = load_dataset("json", data_files={"train": train_jsonl})["train"]
    eval_dataset = load_dataset("json", data_files={"validation": eval_jsonl})[
        "validation"
    ]
    data_collator = VisionDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_steps=-1,
        logging_steps=10,
        save_steps=300,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_total_limit=3,
        save_strategy="steps",
        num_train_epochs=3,
        # evaluation_strategy="steps",
        # eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        processing_class=processor.tokenizer,
    )

    # trainer.train()
    trainer.train(resume_from_checkpoint=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
