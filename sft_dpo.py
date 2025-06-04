import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from trl import DPOTrainer, DPOConfig
import torch
import random

from filter_stem_topic import filter_code_stem_dpo

def preprocess(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

def convert_to_sft(example):
    return {
        "prompt": example["prompt"],
        "response": example["chosen"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to DPO dataset (.jsonl with prompt/chosen/rejected)")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    args = parser.parse_args()

    # Load dataset
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    # Shuffle and split into SFT and DPO subsets
    raw_dataset = raw_dataset.shuffle(seed=42)
    split = raw_dataset.train_test_split(test_size=0.3, seed=42)
    sft_dataset = split["train"].map(convert_to_sft)
    dpo_dataset = split["test"].map(preprocess, remove_columns=split["test"].column_names)

    # Load tokenizer and base model
    base_model = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    # --- Step 1: Supervised Fine-Tuning ---
    print("\n🔧 Starting SFT...")
    sft_output_dir = f"sft_model_{dataset_name}"

    sft_args = TrainingArguments(
        output_dir=sft_output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    def sft_format(example):
        return {
            "input_ids": tokenizer(example["prompt"], truncation=True, max_length=args.max_prompt_length).input_ids,
            "labels": tokenizer(example["response"], truncation=True, max_length=args.max_length).input_ids
        }

    tokenized_sft = sft_dataset.map(sft_format, remove_columns=sft_dataset.column_names)
    trainer = Trainer(model=model, args=sft_args, train_dataset=tokenized_sft)
    trainer.train()

    # --- Step 2: DPO ---
    print("\n🔥 Starting DPO...")
    ref_model = AutoModelForCausalLM.from_pretrained(sft_output_dir, device_map="auto")

    dpo_output_dir = f"dpo_model_{dataset_name}"
    os.makedirs(dpo_output_dir, exist_ok=True)

    dpo_config = DPOConfig(
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=dpo_output_dir,
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    dpo_trainer.train()

    # Push to hub
    print("\n📤 Pushing model to 🤗 Hub...")
    model.push_to_hub(f"{args.hf_username}/{dpo_output_dir}")
    tokenizer.push_to_hub(f"{args.hf_username}/{dpo_output_dir}")
    print("✅ Done!")

if __name__ == "__main__":
    main()