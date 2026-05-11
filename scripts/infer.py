"""Run baseline inference for a single JSONL sample."""
import json

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def build_prompt(question: str, choices: list[str]) -> str:
    if choices:
        options = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
        return (
            "Answer the question based on the image. "
            "Return only the option index.\n"
            f"Question: {question}\n"
            f"Choices:\n{options}\n"
        )
    return (
        "Answer the question based on the image. "
        "Return a short answer.\n"
        f"Question: {question}\n"
    )


def load_sample(jsonl_path: str, index: int) -> dict:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range for {jsonl_path}.")


def main() -> None:
    # Simple configuration for beginners
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    jsonl_path = "data/processed/scienceqa_validation.jsonl"
    sample_index = 0
    max_new_tokens = 64
    temperature = 0.2
    top_p = 0.9

    sample = load_sample(jsonl_path, sample_index)
    image_path = sample.get("image")
    if not image_path:
        raise ValueError("Sample has no image path.")

    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(sample.get("question", ""), sample.get("choices", []))

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    decoded = processor.batch_decode(
        output_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("=== Prompt ===")
    print(prompt)
    print("=== Output ===")
    print(decoded[0])


if __name__ == "__main__":
    main()
