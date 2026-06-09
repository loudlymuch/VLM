"""Gradio web UI entry — 支持多轮对话 + LaTeX 渲染."""
import os
import re
import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model(model_name: str, lora_path: str | None):
    processor = AutoProcessor.from_pretrained(model_name)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = base_model
    if lora_path:
        adapter_config = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print(f"LoRA adapter not found at {lora_path}, using base model.")
    return processor, model


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------
def build_prompt(question: str, choices_str: str, want_reasoning: bool) -> str:
    has_choices = choices_str and choices_str.strip()
    options_text = f"Candidate options: {choices_str}" if has_choices else ""

    instruction = "Analyze the image and solve the question."

    format_instruction = (
        "Your output must strictly follow this format:\n"
        "Reasoning: <your step-by-step analysis>\n"
        "Answer: <the final answer>"
    )
    if not want_reasoning:
        format_instruction = "Your output must strictly follow this format:\nAnswer: <the final answer>"
        instruction += " Be concise."

    return (
        f"{instruction}\n\n"
        f"Question: {question}\n"
        f"{options_text}\n\n"
        f"{format_instruction}"
    )


# ---------------------------------------------------------------------------
# LaTeX → MathJax 兼容转换
# ---------------------------------------------------------------------------
def _process_latex_for_markdown(text: str) -> str:
    """将 `\\(...\\)` / `\\[...\\]` 转为 MathJax 可识别的 `$...$` / `$$...$$`，
    同时保护已有的 `$` / `$$` 不被二次处理。"""
    placeholder_map: dict[str, str] = {}
    counter = [0]

    def _protect(m: re.Match) -> str:
        key = f"__LATEX_PROTECTED_{counter[0]}__"
        counter[0] += 1
        placeholder_map[key] = m.group(0)
        return key

    # 保护已有的 $$…$$（跨行）
    text = re.sub(r"\$\$(.+?)\$\$", _protect, text, flags=re.DOTALL)
    # 保护已有的 $…$（不跨行）
    text = re.sub(r"(?<!\$)\$(.+?)(?<!\$)\$", _protect, text)

    # \(…\) → $…$
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1)}$", text)
    # \[…\] → $$…$$（块级，前后换行避免粘连）
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$$\n{m.group(1)}\n$$", text, flags=re.DOTALL)

    # 还原保护的片段
    for key, value in placeholder_map.items():
        text = text.replace(key, value)

    return text


def _format_output(raw_output: str) -> str:
    """将模型原始输出转为适合 gr.Chatbot 展示的字符串。"""
    output = _process_latex_for_markdown(raw_output)
    # 加粗标签
    output = re.sub(r"^\s*Reasoning:", "\n**Reasoning:**", output, flags=re.MULTILINE)
    output = re.sub(r"^\s*Answer:", "\n**Answer:**", output, flags=re.MULTILINE)
    return output


# ---------------------------------------------------------------------------
# 从原始 messages 列表生成 chatbot 展示用的 (user, bot) 对
# ---------------------------------------------------------------------------
def _extract_question_from_prompt(prompt_text: str) -> str:
    """从完整 prompt 中提取问题文本（支持多行）。"""
    # 匹配 "Question: ..." 直到下一个已知段落标题或结尾
    m = re.search(r"Question:\s*(.+?)(?:\n*(?:Candidate options:|Your output must)|$)",
                  prompt_text, flags=re.DOTALL)
    return m.group(1).strip() if m else prompt_text


def _messages_to_chatbot(messages: list[dict]) -> list[tuple[str | None, str | None]]:
    """将 API 消息列表转为 gr.Chatbot 兼容的 [(user_text, bot_text), ...] 格式。"""
    chatbot_msgs: list[tuple[str | None, str | None]] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        # content 可能是字符串或列表
        if isinstance(content, list):
            text_parts = [c["text"] for c in content if c.get("type") == "text"]
            text = "\n".join(text_parts)
        else:
            text = str(content)

        if role == "user":
            display = _extract_question_from_prompt(text)
            chatbot_msgs.append((display, None))
        elif role == "assistant":
            formatted = _format_output(text)
            if chatbot_msgs and chatbot_msgs[-1][1] is None:
                chatbot_msgs[-1] = (chatbot_msgs[-1][0], formatted)
            else:
                chatbot_msgs.append((None, formatted))
    return chatbot_msgs


# ---------------------------------------------------------------------------
# 核心推理（多轮）
# ---------------------------------------------------------------------------
def chat_turn(
    image,
    question: str,
    want_reasoning: bool,
    choices_str: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    raw_messages: list[dict],   # 完整的 API messages 列表
):
    """执行一轮对话，维护完整的 messages 历史。"""
    # --- 输入校验（不影响对话状态）---
    if image is None:
        err_msgs = [("⚠️", "Please upload an image!")]
        return err_msgs, raw_messages
    if not question:
        err_msgs = [("⚠️", "Please enter a question!")]
        return err_msgs, raw_messages

    prompt = build_prompt(question, choices_str, want_reasoning)

    # --- 构建本轮 user 消息 ---
    if not raw_messages:
        # 第一轮：包含图像
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    else:
        # 后续轮次：仅文本（图像已在历史中）
        user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }

    raw_messages.append(user_msg)

    # --- 生成 ---
    text = processor.apply_chat_template(
        raw_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(raw_messages)
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
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    raw_output = processor.batch_decode(
        output_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # --- 记录 assistant 回复 ---
    raw_messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": raw_output}],
    })

    # --- 更新 chatbot 展示 ---
    chatbot_msgs = _messages_to_chatbot(raw_messages)
    return chatbot_msgs, raw_messages


def clear_chat():
    """清空对话。"""
    return [], []


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
MODEL_NAME = "/mnt/workspace/vlm-model-dir"
LORA_PATH = "/mnt/workspace/VLM/scripts/outputs/checkpoints/qlora_scienceqa/checkpoint-4509"  # 根据实际路径调整

processor, model = load_model(MODEL_NAME, LORA_PATH)

with gr.Blocks(title="VLM Science QA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 VLM Science QA — 多轮对话 Demo")

    with gr.Row():
        # ---- 左侧：图像 + 控制 ----
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 上传图像")
            choices_input = gr.Textbox(
                label="📋 候选项（可选，逗号分隔）",
                placeholder="A) cat, B) dog, C) bird",
            )
            want_reasoning = gr.Checkbox(label="📝 显示推理过程 (Reasoning)", value=True)

            with gr.Accordion("⚙️ 生成参数", open=False):
                max_new_tokens = gr.Slider(16, 512, value=256, step=16, label="Max new tokens")
                temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

        # ---- 右侧：对话区 ----
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 对话",
                height=520,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )
            question_input = gr.Textbox(
                label="✏️ 输入你的问题",
                placeholder="例如：What is shown in this image?",
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 发送", variant="primary")
                clear_btn = gr.Button("🗑️ 清空对话")

    # ---- 状态：存储原始 API messages 列表 ----
    chat_state = gr.State([])

    # ---- 事件绑定 ----
    def _handle_submit(*args):
        chatbot_msgs, new_state = chat_turn(*args)
        return chatbot_msgs, new_state, ""

    submit_btn.click(
        _handle_submit,
        inputs=[
            image_input,
            question_input,
            want_reasoning,
            choices_input,
            max_new_tokens,
            temperature,
            top_p,
            chat_state,
        ],
        outputs=[chatbot, chat_state, question_input],
    )

    question_input.submit(
        _handle_submit,
        inputs=[
            image_input,
            question_input,
            want_reasoning,
            choices_input,
            max_new_tokens,
            temperature,
            top_p,
            chat_state,
        ],
        outputs=[chatbot, chat_state, question_input],
    )

    clear_btn.click(clear_chat, outputs=[chatbot, chat_state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
