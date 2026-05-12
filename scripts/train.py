"""Train the model with a minimal QLoRA setup using Hugging Face Trainer."""
import os

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


def build_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return (
        "Answer the question based on the image. "
        "Return only the option index.\n"
        f"Question: {question}\n"
        f"Choices:\n{options}\n"
    )


class VisionDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = []
        images = []

        for sample in batch:
            image_path = sample.get("image")
            if not image_path:
                continue

            question = sample.get("question", "")
            choices = sample.get("choices", [])
            answer_idx = sample.get("answer")
            if not isinstance(answer_idx, int):
                continue

            prompt = build_prompt(question, choices)
            rationale = sample.get("rationale") or ""
            if rationale:
                answer_text = f"Reasoning: {rationale}\nAnswer: {answer_idx}"
            else:
                answer_text = str(answer_idx)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)
            if not image_inputs:
                continue

            texts.append(text)
            images.append(image_inputs[0])

        inputs = self.processor(
            text=texts,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs


def main() -> None:
    # Simple configuration for beginners
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    train_jsonl = "data/processed/scienceqa_train.jsonl"
    output_dir = "outputs/checkpoints/qlora_scienceqa"
    max_steps = 200
    batch_size = 1
    grad_accum = 4
    lr = 2e-4

    os.makedirs(output_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    dataset = load_dataset("json", data_files={"train": train_jsonl})["train"]
    data_collator = VisionDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=max_steps,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
