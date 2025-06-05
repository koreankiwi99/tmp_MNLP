import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch
import random
from ast import literal_eval
from filter_stem_topic import filter_code_stem_dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to custom DPO dataset (.jsonl)")
    parser.add_argument("--use_public", type=str, default=None,
                        help="Name of a public Hugging Face dataset to use")
    parser.add_argument("--hf_username", type=str, default="koreankiwi99", help="Hugging Face username")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="If set, truncate the dataset to this number of samples.")
    parser.add_argument("--only_singleturn", action="store_true", help="Keep only single-turn examples (context length = 1)")
    args = parser.parse_args()

    if args.use_public:
        print(f"📂 Loading public dataset: {args.use_public}")

        if args.use_public == "nvidia/HelpSteer3":
            raw_dataset = load_dataset(args.use_public, name="preference", split="train")
    
            #raw_dataset = load_dataset(args.use_public, split="preference")
            dataset_name = "HelpSteer3"

            if args.only_singleturn:
                raw_dataset = raw_dataset.filter(lambda x: len(x["context"]) == 1)
                print(f"📊 Filtered single-turn HelpSteer3: {len(raw_dataset)} examples remain")

            def preprocess(example):
                # flatten single-turn context
                turn = example["context"][0]
                prompt = f"{turn['role']}: {turn['content']}"

                if example["overall_preference"] > 0:
                    chosen = example["response1"]
                    rejected = example["response2"]
                else:
                    chosen = example["response2"]
                    rejected = example["response1"]

                return {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }

        else:
            raise ValueError("Only HelpSteer3 is supported for --only_singleturn")

    else:
        if args.data_path is None:
            raise ValueError("You must provide --data_path if not using --use_public.")
        print(f"📂 Loading local dataset: {args.data_path}")
        raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

        def preprocess(example):
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"]
            }

    dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
    dataset = dataset.shuffle(seed=42)

    if args.max_train_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
        print(f"📉 Truncated dataset to {len(dataset)} samples")

    dataset_size = len(dataset)
    model_repo = f"{args.hf_username}/dpo_model_{dataset_name}_{dataset_size}"
    output_dir = "./dpo_model_output"

    print(f"📘 Dataset: {dataset_name}")
    print(f"✅ Loaded {dataset_size} examples")
    print(f"🚀 Model will be pushed to: {model_repo}")

    # Load tokenizer
    base_model = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = None

    # Load model and reference model
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    # DPO training configuration
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=512,
        max_prompt_length=128,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train the model
    trainer.train()

    # Push model/tokenizer to hub
    print("📤 Pushing model to 🤗 Hub...")
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(model_repo)
    print("✅ Done!")


if __name__ == "__main__":
    main()