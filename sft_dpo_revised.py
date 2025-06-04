import argparse
import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from trl import DPOTrainer, DPOConfig

def preprocess_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

def preprocess_sft(example):
    return {
        "prompt": example["prompt"],
        "output": example["chosen"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, help="Name of the HF dataset (e.g. user/dataset)")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--sft_learning_rate", type=float, default=2e-5)
    parser.add_argument("--dpo_learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft_max_length", type=int, default=1024)
    parser.add_argument("--dpo_max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Load dataset
    print(f"📂 Loading dataset: {args.hf_dataset}")
    full_dataset = load_dataset(args.hf_dataset, split="train").shuffle(seed=42)

    split_data = full_dataset.train_test_split(test_size=0.3, seed=42)
    sft_data = split_data["train"].map(preprocess_sft, remove_columns=full_dataset.column_names)
    dpo_data = split_data["test"].map(preprocess_dpo, remove_columns=full_dataset.column_names)

    print(f"✅ SFT samples: {len(sft_data)} | DPO samples: {len(dpo_data)}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    ###### SFT Phase ######
    print("\n🚀 Starting SFT training...")
    sft_model = model
    sft_model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./sft_output",
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.sft_learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        report_to="none"
    )

    def tokenize_sft(batch):
        return tokenizer(batch["prompt"], text_target=batch["output"],
                         padding="max_length", max_length=args.sft_max_length, truncation=True)

    sft_data_tok = sft_data.map(tokenize_sft, batched=True)

    trainer = Trainer(
        model=sft_model,
        args=training_args,
        train_dataset=sft_data_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()

    # Save SFT model
    sft_repo = f"{args.hf_username}/sft_model_{args.hf_dataset.split('/')[-1]}"
    print("\n📤 Pushing SFT model to 🤗 Hub...")
    sft_model.push_to_hub(sft_repo)
    tokenizer.push_to_hub(sft_repo)

    ###### DPO Phase ######
    print("\n🔥 Starting DPO training...")
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    dpo_model = sft_model

    dpo_config = DPOConfig(
        beta=args.beta,
        learning_rate=args.dpo_learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.dpo_max_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        output_dir="./dpo_output",
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
    )

    dpo_trainer = DPOTrainer(
        model=dpo_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dpo_data,
        processing_class=tokenizer,
    )
    dpo_trainer.train()

    # Save DPO model
    dpo_repo = f"{args.hf_username}/dpo_sft_model_revised_{args.hf_dataset.split('/')[-1]}"
    print("\n📤 Pushing DPO model to 🤗 Hub...")
    dpo_trainer.model.push_to_hub(dpo_repo)
    tokenizer.push_to_hub(dpo_repo)
    print("✅ Done!")


if __name__ == "__main__":
    main()