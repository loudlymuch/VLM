"""Inspect dataset sizes saved by HuggingFace `save_to_disk`."""
from pathlib import Path
from datasets import load_from_disk


def main() -> None:
    base = Path("data/raw")
    datasets = ["scienceqa", "mathvista", "chartqa", "textvqa", "docvqa"]

    for name in datasets:
        path = base / name
        if not path.exists():
            print(f"{name}: not found")
            continue

        ds = load_from_disk(str(path))
        if hasattr(ds, "items"):
            sizes = {split: len(d) for split, d in ds.items()}
        else:
            sizes = {"all": len(ds)}
        print(name, sizes)


if __name__ == "__main__":
    main()
