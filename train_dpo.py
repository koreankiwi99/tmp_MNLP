import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to custom DPO dataset (.jsonl)")
    parser.add_argument("--use_public", type=str, default=None,
                    help="Name of a public Hugging Face dataset to use (e.g., 'openbmb/UltraFeedback')")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    args = parser.parse_args()

    # Derived names
    if args.use_public:
        print(f"📂 Loading public dataset: {args.use_public}")

        if args.use_public == "HuggingFaceH4/ultrafeedback_binarized":
            raw_dataset = load_dataset(args.use_public, split="train_prefs")  # or 'train' for full
        
        dataset_name = args.use_public.split("/")[-1]
    
    else:
        if args.data_path is None:
            raise ValueError("You must provide --data_path if not using --use_public.")
        print(f"📂 Loading local dataset: {args.data_path}")
        raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    model_repo = f"{args.hf_username}/dpo_model_{dataset_name}_revision"
    output_dir = f"./{model_repo.replace('/', '_')}"

    print(f"📘 Dataset: {dataset_name}")
    print(f"✅ Loaded {len(raw_dataset)} examples")
    print(f"🚀 Model will be pushed to: {model_repo}")
    
    # Preprocess dataset (keep only needed fields)
    def preprocess(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
    print(dataset[0])

    # Load tokenizer
    base_model = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None

    # Load model and reference model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto"
    )
    
    # DPO training configuration
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-6, #1e-5,
        per_device_train_batch_size=2,  # reduce if OOM
        gradient_accumulation_steps=4,
        max_length=512,
        max_prompt_length=128,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        remove_unused_columns=False,
        fp16=True,                      
        bf16=False,                     
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )

    # Use standard DPOTrainer constructor (NOT from_dataset)
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train the model
    trainer.train()

    # Push model and tokenizer to Hugging Face Hub
    print("📤 Pushing model to 🤗 Hub...")
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(model_repo)
    print("✅ Done!")

if __name__ == "__main__":
    main()