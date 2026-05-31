"""导出 ChartQA/DocVQA 为统一的 JSONL 格式。

教学提示：本脚本目标是把不同数据集“字段对齐”，
让训练脚本只依赖统一字段：image/question/choices/answer/answer_text/rationale/task。
"""
import argparse
import io
import json
import os
from typing import Any

from datasets import load_from_disk
from PIL import Image


DATASETS = {
    # 仅保留需要处理的数据集名称
    "chartqa": "chartqa",
    "docvqa": "docvqa",
}


def _safe_str(value: Any) -> str:
    # 将 None 统一转为空字符串，避免写入 JSONL 时出现 null。
    if value is None:
        return ""
    return str(value)


def _to_pil(image: Any) -> Image.Image | None:
    # 兼容多种图像表示：
    # 1) PIL.Image 直接返回
    # 2) HF Image dict（包含 bytes 或 path）
    # 3) 本地文件路径
    # 返回 None 表示“该样本没有有效图片”。
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path") and os.path.exists(image["path"]):
            return Image.open(image["path"]).convert("RGB")
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image).convert("RGB")
    return None


def _pick_answer(answers: Any) -> str:
    # 多答案时取第一个非空答案，保证训练目标是确定字符串。
    # 这是最简单的做法；更复杂的策略可在这里扩展。
    if isinstance(answers, (list, tuple)):
        for item in answers:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
        return ""
    return _safe_str(answers).strip()


def _resolve_dataset_path(raw_root: str, name: str) -> str:
    # 兼容从仓库根目录或 scripts 目录运行脚本。
    # 这样你可以在任意位置执行 python scripts/xxx.py。
    primary = os.path.join(raw_root, name)
    if os.path.exists(primary):
        return primary
    fallback_root = os.path.join("scripts", "data", "raw")
    fallback = os.path.join(fallback_root, name)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Missing dataset folder: {primary}")


def _write_split(split_ds, split_name: str, dataset_name: str, out_dir: str) -> None:
    # 将一个 split 写成 JSONL，并把图片保存到对应目录。
    # 输出路径示例：data/processed/chartqa_train.jsonl
    # 图片路径示例：data/processed/images/chartqa/train/0.png
    image_root = os.path.join(out_dir, "images", dataset_name, split_name)
    os.makedirs(image_root, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{split_name}.jsonl")

    kept = 0
    skipped_image = 0
    skipped_answer = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(split_ds):
            image = _to_pil(sample.get("image"))
            if image is None:
                # 无图样本跳过，避免训练时崩溃
                skipped_image += 1
                continue

            if dataset_name == "chartqa":
                # ChartQA 的答案字段为 label
                answer_text = _pick_answer(sample.get("label"))
            else:
                # DocVQA 的答案字段为 answers（多候选）
                answer_text = _pick_answer(sample.get("answers"))

            if not answer_text:
                # 无答案样本跳过
                skipped_answer += 1
                continue

            image_path = os.path.join(image_root, f"{idx}.png")
            image.save(image_path)

            question_text = _safe_str(sample.get("question"))
            if not question_text and dataset_name == "chartqa":
                # ChartQA 的问题字段为 query
                question_text = _safe_str(sample.get("query"))

            record = {
                "image": image_path,
                "question": question_text,
                # OCR 类数据通常无固定选项
                "choices": [],
                # 兼容 ScienceQA 格式：无选项时留空
                "answer": None,
                "answer_text": answer_text,
                # 本数据集不包含推理链
                "rationale": "",
                "task": dataset_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"{dataset_name}/{split_name}: kept={kept}, "
        f"skipped_no_image={skipped_image}, skipped_no_answer={skipped_answer}"
    )


def export_dataset(raw_root: str, out_dir: str, dataset_name: str) -> None:
    dataset_path = _resolve_dataset_path(raw_root, dataset_name)
    dataset_dict = load_from_disk(dataset_path)
    if dataset_name == "chartqa":
        # 仅使用 ChartQA 的 train 作为训练集
        # 输出命名为 chartqa_train.jsonl
        split_ds = dataset_dict.get("train")
        if split_ds is None:
            raise ValueError("ChartQA dataset is missing the train split.")
        _write_split(split_ds, "train", dataset_name, out_dir)
        return
    if dataset_name == "docvqa":
        # 仅使用 DocVQA 的 validation 作为训练集
        # 输出命名为 docvqa_train.jsonl（split 名统一为 train）
        split_ds = dataset_dict.get("validation")
        if split_ds is None:
            raise ValueError("DocVQA dataset is missing the validation split.")
        _write_split(split_ds, "train", dataset_name, out_dir)
        return
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ChartQA/DocVQA to JSONL format."
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--datasets",
        default="chartqa,docvqa",
        help="Comma-separated list: chartqa,docvqa",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    selected = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    for name in selected:
        if name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {name}")
        export_dataset(args.raw_root, args.out_dir, name)

    print("OCR datasets JSONL export done.")


if __name__ == "__main__":
    main()
