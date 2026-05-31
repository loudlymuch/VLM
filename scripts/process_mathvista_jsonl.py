"""Export MathVista to unified JSONL format."""
import argparse
import io
import json
import os
from typing import Any

from datasets import load_dataset, load_from_disk
from PIL import Image


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_none(value: Any, none_value: str = "none") -> Any:
    if value is None:
        return none_value
    if isinstance(value, str) and not value.strip():
        return none_value
    return value


def _normalize_choices(value: Any) -> Any:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str) and not value.strip():
        return "none"
    return "none"


def _to_pil(image: Any) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        if image.get("bytes"):
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path") and os.path.exists(image["path"]):
            return Image.open(image["path"]).convert("RGB")
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image).convert("RGB")
    return None


def _extract_meta(sample: dict, split_name: str, image: Image.Image | None) -> dict:
    base = sample.get("metadata") or {}
    meta = {
        "split": split_name,
        "language": _safe_str(base.get("language") or sample.get("language")),
        "img_width": base.get("img_width"),
        "img_height": base.get("img_height"),
        "source": _safe_str(base.get("source") or sample.get("source")),
        "category": _safe_str(base.get("category") or sample.get("category")),
        "task": _safe_str(base.get("task") or sample.get("task")),
        "context": _safe_str(base.get("context") or sample.get("context")),
        "grade": _safe_str(base.get("grade") or sample.get("grade")),
        "skills": base.get("skills") if isinstance(base.get("skills"), list) else [],
    }
    if image is not None:
        width, height = image.size
        meta["img_width"] = width
        meta["img_height"] = height
    return meta


def _write_split(split_ds, split_name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    image_root = os.path.join(out_dir, "images", "mathvista")

    split_dir = os.path.join(image_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mathvista_{split_name}.jsonl")

    kept = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(split_ds):
            image = _to_pil(sample.get("decoded_image") or sample.get("image"))
            if image is None:
                skipped += 1
                continue
            image_path = os.path.join(split_dir, f"{idx}.png")
            image.save(image_path)

            record = {
                "question": _safe_str(sample.get("question")),
                "image": image_path,
                "choices": _normalize_choices(sample.get("choices")),
                "unit": _normalize_none(sample.get("unit")),
                "precision": _normalize_none(sample.get("precision")),
                "answer": _safe_str(sample.get("answer")),
                "question_type": _safe_str(sample.get("question_type")),
                "answer_type": _safe_str(sample.get("answer_type")),
                "pid": _safe_str(sample.get("pid")),
                "metadata": _extract_meta(sample, split_name, image),
                "query": _safe_str(sample.get("query")),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{split_name}: kept={kept}, skipped_no_image={skipped}")


def export_mathvista_jsonl(dataset_dict, out_dir: str) -> None:
    for split_name, split_ds in dataset_dict.items():
        _write_split(split_ds, split_name, out_dir)


def export_mathvista_testmini_split(dataset_dict, out_dir: str, train_ratio: float, seed: int) -> None:
    if "testmini" not in dataset_dict:
        raise ValueError("MathVista dataset is missing the testmini split.")
    split_ds = dataset_dict["testmini"]
    split = split_ds.train_test_split(test_size=1.0 - train_ratio, seed=seed, shuffle=True)
    _write_split(split["train"], "train", out_dir)
    _write_split(split["test"], "validation", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MathVista to JSONL format.")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw/mathvista")
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--mode", choices=["testmini-split", "all"], default="testmini-split")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.from_hf and os.path.exists(args.raw_dir):
        dataset_mv = load_from_disk(args.raw_dir)
    else:
        dataset_mv = load_dataset("AI4Math/MathVista")

    if args.mode == "all":
        export_mathvista_jsonl(dataset_mv, args.out_dir)
    else:
        export_mathvista_testmini_split(
            dataset_mv,
            args.out_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    print("MathVista JSONL export done.")


if __name__ == "__main__":
    main()
