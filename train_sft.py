import argparse
import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi, create_repo


def preprocess_sft(example):
    prompt = example["instruction"]
    output = example["output"]
    return {"text": f"{prompt.strip()}\n\n{output.strip()}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to SFT config JSON file")
    parser.add_argument("--sft_dataset_name", type=str, required=True, help="Hugging Face dataset name (instruction-output format)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B-Base", help="Base model name or path")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    args = parser.parse_args()

    # Load dataset
    print(f"📂 Loading SFT dataset: {args.sft_dataset_name}")
    dataset = load_dataset(args.sft_dataset_name, split="train")
    dataset = dataset.map(preprocess_sft, remove_columns=dataset.column_names)

    # Load config
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    max_length = config_dict.pop("max_length", 1024)  # Used only for tokenization
    output_dir = config_dict.get("output_dir")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Tokenize dataset
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)

    # Define output and repo
    config_name = os.path.basename(args.config_path).replace('.json', '')
    dataset_name = args.sft_dataset_name.split('/')[-1]
    if not output_dir:
        output_dir = f"./{args.hf_username}_sft_{config_name}_{dataset_name}"
        config_dict["output_dir"] = output_dir
    repo_id = f"{args.hf_username}/sft_model_{config_name}_{dataset_name}"

    # Setup trainer
    training_args = TrainingArguments(**config_dict, report_to="none")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("🚀 Starting SFT training...")
    trainer.train()

    print("💾 Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "sft_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"🌐 Uploading to Hugging Face Hub: {repo_id}")
    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        repo_type="model"
    )

    print("✅ SFT model training and upload complete!")


if __name__ == "__main__":
    main()