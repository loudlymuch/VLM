"""Evaluate fine-tuned LoRA accuracy on a JSONL dataset."""
import json

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def build_prompt(question: str) -> str:
    return (
        "Answer the question based on the image. "
        "Return in the format: Reasoning: ... Answer: ...\n"
        f"Question: {question}\n"
    )


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).strip()


def extract_answer_text(pred_text: str) -> str:
    if not pred_text:
        return ""
    lowered = pred_text.lower()
    marker = "answer:"
    pos = lowered.find(marker)
    if pos == -1:
        return pred_text
    return pred_text[pos + len(marker) :]


def is_correct(pred_text: str, answer_text: str) -> bool:
    if not pred_text or not answer_text:
        return False
    extracted = extract_answer_text(pred_text)
    pred_norm = normalize_text(extracted)
    ans_norm = normalize_text(answer_text)
    return ans_norm in pred_norm


def main() -> None:
    # Simple configuration for beginners
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    jsonl_path = "data/processed/scienceqa_validation.jsonl"
    jsonl_train_path = "data/processed/scienceqa_train.jsonl"
    lora_path = "outputs/checkpoints/qlora_scienceqa"
    max_samples = 1000
    max_new_tokens = 512
    temperature = 0.2
    top_p = 0.9

    processor = AutoProcessor.from_pretrained(model_name)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)

    total = 0
    correct = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and total >= max_samples:
                break

            sample = json.loads(line)
            image_path = sample.get("image")
            if not image_path:
                continue

            question = sample.get("question", "")
            choices = sample.get("choices", [])
            answer_idx = sample.get("answer")
            answer_text = sample.get("answer_text") or ""
            if not answer_text and isinstance(answer_idx, int) and choices:
                if 0 <= answer_idx < len(choices):
                    answer_text = str(choices[answer_idx])
            if not answer_text:
                continue

            prompt = build_prompt(question)
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
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            output_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            decoded = processor.batch_decode(
                output_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            pred_text = decoded[0]

            total += 1
            if is_correct(pred_text, answer_text):
                correct += 1

            if total % 20 == 0:
                acc = correct / total if total else 0.0
                print(f"Progress: {total}, acc={acc:.3f}")

    acc = correct / total if total else 0.0
    print("=== Result ===")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
