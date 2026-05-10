"""Prepare datasets for training and evaluation."""
import os
from datasets import load_dataset

def main() -> None:
    # 定义保存路径
    raw_data_path = "data/raw"
    os.makedirs(raw_data_path, exist_ok=True)

    print("开始下载 ScienceQA...")
    # ScienceQA 官方数据集 ID
    # derek-thomas/ScienceQA 是比较常用的版本
    dataset_sqa = load_dataset("derek-thomas/ScienceQA")
    dataset_sqa.save_to_disk(os.path.join(raw_data_path, "scienceqa"))
    print("ScienceQA 下载完成！")

    print("\n开始下载 MathVista...")
    # MathVista 官方数据集 ID
    # 注意：MathVista 比较大，包含很多高分辨率图片
    dataset_mv = load_dataset("AI4Math/MathVista")
    dataset_mv.save_to_disk(os.path.join(raw_data_path, "mathvista"))
    print("MathVista 下载完成！")

    dataset_chart = load_dataset("lmms-lab/ChartQA")
    dataset_chart.save_to_disk(os.path.join(raw_data_path, "chartqa"))
    print("ChartQA 下载完成！")

    dataset_textvqa = load_dataset("lmms-lab/textvqa")
    dataset_textvqa.save_to_disk(os.path.join(raw_data_path, "textvqa"))
    print("TextVQA 下载完成！")

    dataset_docvqa = load_dataset("lmms-lab/DocVQA", "DocVQA")
    dataset_docvqa.save_to_disk(os.path.join(raw_data_path, "docvqa"))
    print("DocVQA 下载完成！")

    # dataset_vqa = load_dataset("lmms-lab/VQAv2")
    # dataset_vqa.save_to_disk(os.path.join(raw_data_path, "vqa"))
    # print("VQAv2 下载完成！")

if __name__ == "__main__":
    main()
