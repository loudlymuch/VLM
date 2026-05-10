"""Export ScienceQA to unified JSONL format."""
import json
import os
from datasets import load_dataset


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def export_scienceqa_jsonl(dataset_dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    image_root = os.path.join(out_dir, "images")

    for split_name, split_ds in dataset_dict.items():
        split_dir = os.path.join(image_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"scienceqa_{split_name}.jsonl")

        with open(out_path, "w", encoding="utf-8") as f:
            for idx, sample in enumerate(split_ds):
                image = sample.get("image")
                image_path = ""
                if image is not None:
                    image_path = os.path.join(split_dir, f"{idx}.png")
                    image.save(image_path)

                choices = sample.get("choices") or []
                answer_idx = sample.get("answer")
                answer_text = ""
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                    answer_text = _safe_str(choices[answer_idx])

                lecture = _safe_str(sample.get("lecture"))
                solution = _safe_str(sample.get("solution"))
                rationale = ""
                if lecture or solution:
                    rationale = "\n".join([line for line in [lecture, solution] if line])

                record = {
                    "image": image_path,
                    "question": _safe_str(sample.get("question")),
                    "choices": choices,
                    "answer": answer_idx,
                    "answer_text": answer_text,
                    "rationale": rationale,
                    "task": "scienceqa",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    dataset_sqa = load_dataset("derek-thomas/ScienceQA")
    export_scienceqa_jsonl(dataset_sqa, "data/processed")
    print("ScienceQA JSONL export done.")


if __name__ == "__main__":
    main()
