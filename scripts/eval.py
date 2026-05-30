"""Evaluate fine-tuned LoRA accuracy on a JSONL dataset.

Supports both multiple-choice (ScienceQA) and open-ended questions.
Primary metric: Exact Match (EM) after answer extraction + normalization.
"""

import json

import argparse
import re
from difflib import SequenceMatcher

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(pred_text: str) -> str:
    """Pull the answer portion from a model-generated string.

    Tries multiple patterns in order; returns the first non-empty match.
    Falls back to the original text if no pattern matches.
    """
    if not pred_text:
        return ""

    # Helper: truncate to first sentence after a marker
    def _first_sentence(text: str) -> str:
        m = re.search(r"[.。；;！!？?\n]", text)
        if m:
            return text[:m.start()].strip()
        return text

    # Primary: trained format
    for marker in ["answer:", "Answer:", "ANSWER:"]:
        pos = pred_text.lower().find(marker.lower())
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker):].strip())

    # Chinese markers (for Chinese-language questions)
    for marker in ["答案是", "答案为", "答案：", "回答："]:
        pos = pred_text.find(marker)
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker):].strip())

    # English natural-language markers
    for marker in ["The answer is", "the answer is"]:
        pos = pred_text.find(marker)
        if pos != -1:
            return _first_sentence(pred_text[pos + len(marker):].strip())

    # Last resort: take the last non-empty line (often contains the answer)
    lines = [l.strip() for l in pred_text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return pred_text


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalise text for comparison.

    - lowercase
    - collapse whitespace
    - strip leading/trailing punctuation that isn't semantically meaningful
    - normalise number formats (trailing ".0" on integers)
    """
    text = text.strip().lower()
    text = " ".join(text.split())
    # Strip leading/trailing punctuation except minus sign
    text = text.strip(".,;:!?\"'()[]{} \t\n\r")
    return text


def fuzzy_match(a: str, b: str) -> float:
    """SequenceMatcher ratio in [0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------

def is_exact_match(pred: str, target: str) -> bool:
    """Normalised exact-match."""
    return normalize(pred) == normalize(target)


def is_contained(pred: str, target: str) -> bool:
    """Check whether the normalised target is a *word-bounded* substring of pred.

    Uses word-boundary checks to avoid "cat" matching "category".
    """
    p = normalize(pred)
    t = normalize(target)
    if not t or not p:
        return False
    # Check all occurrences for a whole-word match
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
    return False


def best_option_match(pred_answer: str, choices: list) -> int | None:
    """Return the index of the choice that best matches *pred_answer*.

    Uses exact-match first, then fuzzy matching as fallback.
    Returns None if no choice is similar enough (threshold 0.5).
    """
    if not choices:
        return None

    best_idx = None
    best_score = 0.0

    for i, choice in enumerate(choices):
        choice_str = str(choice).strip()
        # Exact after normalisation → instant win
        if normalize(pred_answer) == normalize(choice_str):
            return i
        score = fuzzy_match(pred_answer, choice_str)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx if best_score >= 0.5 else None


# ---------------------------------------------------------------------------
# Prompt builder (mirrors train.py)
# ---------------------------------------------------------------------------

def build_prompt(question: str, choices: list = None) -> str:
    has_choices = choices is not None and len(choices) > 0
    
    if has_choices:
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


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned VLM on JSONL data.")
    parser.add_argument("--model", default="/mnt/workspace/vlm-model-dir")
    parser.add_argument("--jsonl", default="data/processed/scienceqa_validation.jsonl")
    parser.add_argument("--lora", default="outputs/checkpoints/qlora_scienceqa/checkpoint-2334")
    parser.add_argument("--test-mode", choices=["with_options", "no_options"], default="with_options", 
                        help="Evaluate with or without candidate options.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--verbose", "-v", action="store_true", help="print every wrong case")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora, local_files_only=True)

    model.eval()
    # ---- Load all samples into memory so we can report stats by type ----
    samples: list[dict] = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if args.max_samples and len(samples) >= args.max_samples:
                break

    total = 0
    correct_em = 0          # normalised exact match
    correct_contain = 0     # word-bounded containment
    correct_option = 0      # best-choice-index match (MC only)
    mc_total = 0

    for sample in samples:
        image_path = sample.get("image")
        if not image_path:
            continue

        question = sample.get("question", "")
        choices = sample.get("choices") or []
        answer_idx = sample.get("answer")       # int index (MC)
        answer_text = sample.get("answer_text") or ""

        # Reconstruct answer_text from choices if missing
        if not answer_text and isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
            answer_text = str(choices[answer_idx])
        if not answer_text:
            continue

        # ---- Inference ----
        # 针对混合训练的评估逻辑
        if args.test_mode == "with_options":
            eval_choices = choices
        else:
            eval_choices = None # 强行不给选项，测试模型的裸推理能力

        prompt = build_prompt(question, eval_choices)
        
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

        # ---- Scoring ----
        total += 1
        em_ok = is_exact_match(pred_answer, answer_text)
        contain_ok = is_contained(pred_answer, answer_text)
        if em_ok:
            correct_em += 1
        if contain_ok:
            correct_contain += 1

        # Option-based match (only meaningful for MC with choices)
        option_ok = False
        if choices and isinstance(answer_idx, int):
            mc_total += 1
            matched_idx = best_option_match(pred_answer, choices)
            if matched_idx is not None and matched_idx == answer_idx:
                correct_option += 1
                option_ok = True

        # ---- Verbose logging ----
        if args.verbose and not em_ok:
            print(f"\n{'='*60}")
            print(f"[{total}] EM=FALSE  contain={contain_ok}  option={option_ok}")
            print(f"  Q: {question[:120]}")
            print(f"  GT answer_text: {repr(answer_text)}")
            print(f"  GT answer_idx : {answer_idx}")
            print(f"  Choices       : {choices}")
            print(f"  Pred raw      : {pred_text[:200]}")
            print(f"  Pred extracted: {repr(pred_answer)}")
            print(f"  Pred norm     : {repr(normalize(pred_answer))}")
            print(f"  GT norm       : {repr(normalize(answer_text))}")

        if total % 20 == 0:
            print(f"[{total}/{len(samples)}] "
                  f"EM={correct_em / total:.3f}  "
                  f"contain={correct_contain / total:.3f}  "
                  f"option_acc={correct_option / max(mc_total, 1):.3f}")

    # ---- Final report ----
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"  Total samples          : {total}")
    print(f"  MC samples             : {mc_total}")
    print(f"  EM (exact match)       : {correct_em}/{total} = {correct_em / total:.4f}")
    print(f"  Contain (word-bound)   : {correct_contain}/{total} = {correct_contain / total:.4f}")
    if mc_total:
        print(f"  Option-index accuracy  : {correct_option}/{mc_total} = {correct_option / mc_total:.4f}")
    print()

    # Print per-metric summary for easy copying into report
    print("=== TL;DR ===")
    print(f"EM={correct_em / total:.4f}  "
          f"option_acc={correct_option / max(mc_total, 1):.4f}  "
          f"contain={correct_contain / total:.4f}")


if __name__ == "__main__":
    main()
