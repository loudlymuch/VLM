"""
生成人工评估材料：分层抽样覆盖全部 4 个数据集，选择题 + 生成式（开放题）两种模式。

抽样策略（默认 ~80 条，覆盖所有维度）：
  ScienceQA     → 15 条带选项（选择题）+ 10 条隐藏选项（生成式）
  MathVista     → 10 条带选项（选择题）+ 10 条生成式（开放题）
  ChartQA       → 15 条生成式（天然开放题，无选项）
  DocVQA        → 15 条生成式（天然开放题，无选项）
                          ─────────
                           共 75 条

产出文件：
  outputs/human_eval/human_eval_samples.md   → 供阅读评分的 Markdown 文件（图片+输出+评分表）
  outputs/human_eval/human_eval_scores.csv   → 供 Excel 填分的 CSV 模板

用法：
  python scripts/gen_human_eval.py                           # 默认分层抽样
  python scripts/gen_human_eval.py --total 50                # 总数约 50 条（按比例缩放）
  python scripts/gen_human_eval.py --lora outputs/checkpoints/qlora_multitask/checkpoint-900
"""

import argparse
import csv
import json
import os
import random
import sys

import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================================
# 默认抽样方案
#   key: JSONL 文件名（不含 .jsonl）
#   with_options: 带选项推理的抽样数（0 表示该数据集没有选择题）
#   without_options: 隐藏选项/开放题的抽样数
#   note: 数据集的简短描述
# ============================================================================

DEFAULT_PLAN = [
    {"key": "scienceqa_validation",  "with": 15, "without": 10, "note": "ScienceQA (科学问答)"},
    {"key": "mathvista_validation",  "with": 10, "without": 10, "note": "MathVista (数学视觉)"},
    {"key": "chartqa_train",         "with":  0, "without": 15, "note": "ChartQA (图表问答)"},
    {"key": "docvqa_train",          "with":  0, "without": 15, "note": "DocVQA (文档问答)"},
]


# ============================================================================
# 工具函数（与 train.py 保持一致）
# ============================================================================

def normalize_sample(sample: dict) -> dict | None:
    """统一字段格式，同 train.py。"""
    image_path = sample.get("image")
    if not image_path:
        return None

    question = sample.get("question", "") or ""
    choices = sample.get("choices") or []
    if isinstance(choices, str):
        choices = []

    answer_text = sample.get("answer_text") or ""
    answer_idx = sample.get("answer")

    if not answer_text:
        if isinstance(answer_idx, int) and choices and 0 <= answer_idx < len(choices):
            answer_text = str(choices[answer_idx])
        elif isinstance(answer_idx, str) and answer_idx.strip():
            answer_text = answer_idx.strip()
        elif answer_idx is not None:
            answer_text = str(answer_idx).strip()

    if not answer_text:
        return None

    rationale = sample.get("rationale") or ""

    return {
        "image": image_path,
        "question": question,
        "choices": choices,
        "answer": answer_idx if isinstance(answer_idx, int) else None,
        "answer_text": answer_text,
        "rationale": rationale,
    }


def build_prompt(question: str, choices: list | None) -> str:
    """构建强制推理格式的 prompt。"""
    has_choices = choices is not None and len(choices) > 0

    if has_choices:
        options_text = "Candidate options: " + ", ".join(choices)
        instruction = (
            f"Considering the following options: [{options_text}], "
            "analyze the image and solve the question."
        )
    else:
        instruction = (
            "Analyze the image and provide a direct answer to the following scientific question."
        )

    return (
        f"{instruction}\n\n"
        f"Question: {question}\n\n"
        "Your output must strictly follow this format:\n"
        "Reasoning: <your step-by-step scientific analysis>\n"
        "Answer: <the specific result or term>\n\n"
        "IMPORTANT: You MUST include the Reasoning section. Do NOT skip it. "
        "Even if the answer seems obvious, explain your thought process first."
    )


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_processor(model_path: str, lora_path: str):
    """加载 4-bit 量化模型 + LoRA adapter。"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, processor


# ============================================================================
# 单样本推理
# ============================================================================

@torch.no_grad()
def infer_one(
    sample: dict,
    model,
    processor,
    with_options: bool,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """对单个样本推理，返回模型生成的完整文本。"""
    choices = sample["choices"] if with_options else []
    prompt = build_prompt(sample["question"], choices)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
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
    return decoded[0]


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="生成人工评估材料（分层抽样）")
    parser.add_argument("--model", default="/mnt/workspace/vlm-model-dir",
                        help="基础模型路径")
    parser.add_argument("--lora", default=None,
                        help="LoRA checkpoint 路径（留空自动查找）")
    parser.add_argument("--total", type=int, default=None,
                        help="目标总抽样数（按默认比例缩放，默认 ~75）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output-dir", default="outputs/human_eval",
                        help="输出目录")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成 token 数")
    args = parser.parse_args()

    # ---- 自动查找最新 checkpoint ----
    if args.lora is None:
        import glob
        candidates = sorted(glob.glob(
            "outputs/checkpoints/qlora_multitask/checkpoint-*"))
        if candidates:
            args.lora = candidates[-1]
            print(f"自动选择 LoRA checkpoint: {args.lora}")
        else:
            print("错误: 未找到 LoRA checkpoint，请用 --lora 指定路径")
            sys.exit(1)

    # ---- 根据 --total 缩放抽样数 ----
    plan = []
    for entry in DEFAULT_PLAN:
        plan.append({
            "key": entry["key"],
            "file": f"data/processed/{entry['key']}.jsonl",
            "with": entry["with"],
            "without": entry["without"],
            "note": entry["note"],
        })

    if args.total is not None:
        default_total = sum(e["with"] + e["without"] for e in plan)
        scale = args.total / default_total
        for e in plan:
            e["with"] = max(0, round(e["with"] * scale))
            e["without"] = max(0, round(e["without"] * scale))

    total_planned = sum(e["with"] + e["without"] for e in plan)
    print(f"抽样方案: 共 {total_planned} 条")

    # ---- 创建输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)
    image_out_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_out_dir, exist_ok=True)

    # ---- 设置随机种子 ----
    random.seed(args.seed)

    # ---- 加载模型 ----
    print("正在加载模型...")
    model, processor = load_model_and_processor(args.model, args.lora)
    print("模型加载完成。")

    # ---- 逐数据集抽样并推理 ----
    all_results = []
    sample_id = 0

    for ds in plan:
        jsonl_path = ds["file"]
        if not os.path.exists(jsonl_path):
            print(f"  [WARN] 文件不存在，跳过: {jsonl_path}")
            continue

        # 加载并归一化全部样本
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                normed = normalize_sample(json.loads(line))
                if normed is not None:
                    samples.append(normed)

        if len(samples) == 0:
            print(f"  [WARN] {ds['key']}: 无有效样本")
            continue

        # 拆分为"有选项"和"无选项"候选池
        has_c = [s for s in samples if len(s["choices"]) > 0]
        no_c  = [s for s in samples if len(s["choices"]) == 0]

        # ---- 带选项的抽样（从有选项池中取） ----
        n_with = min(ds["with"], len(has_c))
        chosen_with = random.sample(has_c, n_with) if n_with > 0 else []

        # ---- 不带选项的抽样：优先从无选项池取，不够再从有选项池补（隐藏选项） ----
        n_without = ds["without"]
        chosen_without = []
        if n_without > 0:
            from_no_c = min(n_without, len(no_c))
            chosen_without.extend(random.sample(no_c, from_no_c))
            # 不够的从有选项池补（注意排除已被 with 抽走的）
            remainder = n_without - from_no_c
            if remainder > 0:
                available = [s for s in has_c if s not in chosen_with]
                remainder = min(remainder, len(available))
                chosen_without.extend(random.sample(available, remainder))

        print(f"\n{'='*60}")
        print(f"{ds['note']} ({ds['key']})")
        print(f"  总样本 {len(samples)} | 有选项池 {len(has_c)} | 无选项池 {len(no_c)}")
        print(f"  带选项抽 {len(chosen_with)} 条 | 生成式抽 {len(chosen_without)} 条")
        print(f"{'='*60}")

        # ---- 推理 ----
        for mode, chosen in [("MC", chosen_with), ("Gen", chosen_without)]:
            with_options = (mode == "MC")
            mode_label = "选择题模式" if with_options else "生成式（无选项）"

            for i, sample in enumerate(chosen):
                sample_id += 1
                print(f"  [{sample_id}] {ds['key']}/{mode} "
                      f"({i+1}/{len(chosen)}) ...", end=" ", flush=True)

                output = infer_one(
                    sample, model, processor,
                    with_options=with_options,
                    max_new_tokens=args.max_tokens,
                )

                # 复制图片到输出目录
                src_img = sample["image"]
                img_ext = os.path.splitext(src_img)[1] or ".png"
                dst_img = os.path.join(
                    image_out_dir, f"sample_{sample_id:03d}{img_ext}")
                if os.path.exists(src_img):
                    try:
                        Image.open(src_img).save(dst_img)
                    except Exception:
                        dst_img = src_img
                else:
                    dst_img = src_img

                has_reasoning = ("reasoning:" in output.lower()
                                 or "Reasoning:" in output)
                has_answer = ("answer:" in output.lower()
                              or "Answer:" in output)

                result = {
                    "id": sample_id,
                    "dataset": ds["key"],
                    "mode": mode,
                    "mode_label": mode_label,
                    "image": dst_img,
                    "question": sample["question"],
                    "choices_given": (
                        ", ".join(sample["choices"]) if with_options and sample["choices"]
                        else "(未提供选项)"),
                    "choices_all": ", ".join(sample["choices"]) if sample["choices"] else "(无选项)",
                    "answer_gt": sample["answer_text"],
                    "rationale_gt": sample["rationale"][:300] if sample["rationale"] else "",
                    "model_output": output,
                    "has_reasoning": has_reasoning,
                    "has_answer": has_answer,
                }
                all_results.append(result)
                print("完成")

    # =====================================================================
    # 生成 Markdown 评估表
    # =====================================================================
    md_path = os.path.join(args.output_dir, "human_eval_samples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 人工评估材料\n\n")
        f.write(f"> 总样本数: {len(all_results)}  |  随机种子: {args.seed}\n\n")

        # 抽样概况
        f.write("## 抽样概况\n\n")
        f.write("| 数据集 | 选择题模式 | 生成式模式 |\n")
        f.write("|--------|-----------|----------|\n")
        for ds in plan:
            n_mc = sum(1 for r in all_results
                       if r["dataset"] == ds["key"] and r["mode"] == "MC")
            n_gen = sum(1 for r in all_results
                        if r["dataset"] == ds["key"] and r["mode"] == "Gen")
            if n_mc + n_gen > 0:
                f.write(f"| {ds['note']} | {n_mc} | {n_gen} |\n")
        f.write("\n---\n\n")

        # 评分维度说明
        f.write("## 评分维度\n\n")
        f.write("每题请在以下 4 个维度上打分。**先不要展开参考答案**，评完再对照。\n\n")
        f.write("| 维度 | 说明 | 分值 |\n")
        f.write("|------|------|------|\n")
        f.write("| **推理正确性** | 推理链逻辑是否严密，有无知识性错误 | 1-5 |\n")
        f.write("| **推理完整性** | 关键步骤是否缺失、是否跳跃 | 1-5 |\n")
        f.write("| **答案正确性** | 最终答案是否准确（参考 GT） | 0/1 |\n")
        f.write("| **表达清晰度** | 语言通顺、术语准确、无冗余 | 1-5 |\n\n")
        f.write("> 注意：如果模型没有输出推理链，推理正确性和完整性评 **N/A**。\n\n")
        f.write("---\n\n")

        # 逐样本
        for r in all_results:
            f.write(f"## 样本 {r['id']}  |  "
                    f"{r['dataset']}  |  "
                    f"**{r['mode_label']}**\n\n")
            f.write(f"**图片**: `{r['image']}`\n\n")
            f.write(f"**问题**: {r['question']}\n\n")
            f.write(f"**给定选项**: {r['choices_given']}\n\n")
            f.write("**模型输出**:\n\n")
            f.write(f"```\n{r['model_output']}\n```\n\n")

            f.write("### 评分\n\n")
            f.write("| 维度 | 评分 | 备注 |\n")
            f.write("|------|------|------|\n")
            f.write("| 推理正确性 (1-5 / N/A) |  |  |\n")
            f.write("| 推理完整性 (1-5 / N/A) |  |  |\n")
            f.write("| 答案正确性 (0/1) |  |  |\n")
            f.write("| 表达清晰度 (1-5) |  |  |\n\n")

            # 折叠的参考答案
            f.write("<details>\n<summary>📝 参考答案 & 元信息（评完后展开）</summary>\n\n")
            f.write(f"- **正确答案 (GT)**: {r['answer_gt']}\n")
            if r['rationale_gt']:
                f.write(f"- **参考推理 (GT)**: {r['rationale_gt']}\n")
            f.write(f"- **全部选项**: {r['choices_all']}\n")
            f.write(f"- **输出含 Reasoning**: {'✅' if r['has_reasoning'] else '❌'}  |  ")
            f.write(f"**输出含 Answer**: {'✅' if r['has_answer'] else '❌'}\n")
            f.write("\n</details>\n\n")
            f.write("---\n\n")

    print(f"\nMarkdown 评估表已保存: {md_path}")

    # =====================================================================
    # 生成 CSV 评分模板
    # =====================================================================
    csv_path = os.path.join(args.output_dir, "human_eval_scores.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "样本ID", "数据集", "模式", "问题", "给定选项",
            "模型输出", "正确答案(GT)",
            "推理正确性(1-5)", "推理完整性(1-5)",
            "答案正确性(0/1)", "表达清晰度(1-5)",
            "备注"
        ])
        for r in all_results:
            writer.writerow([
                r["id"], r["dataset"], r["mode_label"],
                r["question"], r["choices_given"],
                r["model_output"], r["answer_gt"],
                "", "", "", "", ""   # 待填
            ])

    print(f"CSV 评分模板已保存: {csv_path}")

    # =====================================================================
    # 统计摘要
    # =====================================================================
    n_with_reasoning = sum(1 for r in all_results if r["has_reasoning"])
    n_with_answer = sum(1 for r in all_results if r["has_answer"])
    n_mc = sum(1 for r in all_results if r["mode"] == "MC")
    n_gen = sum(1 for r in all_results if r["mode"] == "Gen")

    print(f"\n{'='*60}")
    print(f"生成完毕！")
    print(f"  总样本:       {len(all_results)}")
    print(f"  选择题模式:   {n_mc}")
    print(f"  生成式模式:   {n_gen}")
    print(f"  含推理链:     {n_with_reasoning}/{len(all_results)}")
    print(f"  含答案标记:   {n_with_answer}/{len(all_results)}")
    print(f"  输出目录:     {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
