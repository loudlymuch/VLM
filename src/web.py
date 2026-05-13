"""Gradio web UI entry."""
import os
import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel


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


def build_prompt(question: str, want_reasoning: bool) -> str:
	prompt = (
		"Answer the question based on the image. "
		"Return in the format: Reasoning: ... Answer: ...\n"
		f"Question: {question}\n"
	)
	if want_reasoning:
		prompt += "Explain your reasoning briefly."
	return prompt


def infer(image, question, want_reasoning, max_new_tokens, temperature, top_p):
	if image is None:
		return "Please upload an image."
	if not question:
		return "Please enter a question."

	prompt = build_prompt(question, want_reasoning)
	messages = [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": image},
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
	return decoded[0]


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_PATH = "scripts/outputs/checkpoints/qlora_scienceqa"

processor, model = load_model(MODEL_NAME, LORA_PATH)


with gr.Blocks(title="VLM Science QA Demo") as demo:
	gr.Markdown("# VLM Science QA Demo")

	with gr.Row():
		image_input = gr.Image(type="pil", label="Image")
		with gr.Column():
			question_input = gr.Textbox(label="Question")
			want_reasoning = gr.Checkbox(label="Show reasoning", value=False)
			max_new_tokens = gr.Slider(16, 256, value=64, step=8, label="Max new tokens")
			temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
			top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
			submit_btn = gr.Button("Run")

	output_text = gr.Textbox(label="Output")

	submit_btn.click(
		infer,
		inputs=[
			image_input,
			question_input,
			want_reasoning,
			max_new_tokens,
			temperature,
			top_p,
		],
		outputs=output_text,
	)


if __name__ == "__main__":
	demo.launch()
