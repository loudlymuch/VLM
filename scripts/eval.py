"""Evaluate baseline accuracy on a JSONL dataset."""
import json

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def build_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return (
        "Answer the question based on the image. "
        "Return only the option index.\n"
        f"Question: {question}\n"
        f"Choices:\n{options}\n"
    )


def parse_index(text: str) -> int:
    digits = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    cleaned = text.strip()
    if not cleaned:
        return -1
    digits = ""
    for ch in cleaned:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return -1
    return int(digits)


def main() -> None:
    # Simple configuration for beginners
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    jsonl_path = "data/processed/scienceqa_validation.jsonl"
    max_samples = 1000
    max_new_tokens = 32
    temperature = 0.2
    top_p = 0.9

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

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
            if not isinstance(answer_idx, int):
                continue

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
            pred_idx = parse_index(decoded[0])

            total += 1
            if pred_idx == answer_idx:
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
