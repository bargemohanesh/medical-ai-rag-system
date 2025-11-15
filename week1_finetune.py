import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

print("="*60)
print("WEEK 1: FINE-TUNING LLAMA 3.1 8B ON MEDICAL DATA")
print("="*60)

model_path = "/mnt/c/Users/Mohanesh/models/Llama-3.1-8B-Instruct"
print(f"\n1. Loading model from: {model_path}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)

print("   ✅ Model loaded in 4-bit")

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print("\n2. Model Configuration:")
model.print_trainable_parameters()

print("\n3. Loading dataset...")
dataset = load_from_disk("./data/medmcqa_10k")

def format_example(example):
    question = example["question"]
    options = [example[f"op{i}"] for i in ["a", "b", "c", "d"]]
    correct_idx = example["cop"]
    answer = options[correct_idx]
    explanation = example.get("exp", "")
    full_answer = f"{answer}. {explanation}" if explanation else answer
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert medical AI assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{full_answer}<|eot_id|>"""
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"   ✅ Formatted {len(dataset)} examples")

training_args = TrainingArguments(
    output_dir="./week1_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("\n4. Starting training...")
print(f"   Total steps: ~{len(dataset) // 16 * 3}")
print(f"   Expected time: ~2-3 hours")
print(f"   VRAM usage: ~10-12GB / 16GB\n")

trainer.train()

output_dir = "./week1_model_final"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n{'='*60}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Model saved to: {output_dir}")
