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
    args = parser.parse_args()

    # Derived names
    if args.use_public:
        print(f"📂 Loading public dataset: {args.use_public}")

        if args.use_public == "HuggingFaceH4/ultrafeedback_binarized":
            raw_dataset = load_dataset(args.use_public, split="train_prefs")  # or 'train' for full
        
        elif args.use_public == "HuggingFaceH4/stack-exchange-preferences":
            from filter_stem_topic import is_code_stem_prompt

            def is_valid_stem_example(example):
                # Check: 2+ scored answers
                scored_answers = [a for a in example["answers"] if "pm_score" in a and isinstance(a["pm_score"], (int, float))]
                if len(scored_answers) < 2:
                    return False

                # Check: STEM-related prompt
                return is_code_stem_prompt(example.get("question", ""))

            # Load and filter in a single pass
            raw_dataset = load_dataset(args.use_public, split="train").filter(is_valid_stem_example)
        
        elif args.use_public == "argilla/ultrafeedback-binarized-preferences-cleaned":
            ds = load_dataset(args.use_public, split="train")
            raw_dataset = filter_code_stem_dpo(ds)

        else:
            raw_dataset = load_dataset(args.use_public, split="train")

        dataset_name = args.use_public.split("/")[-1]
    
    else:
        if args.data_path is None:
            raise ValueError("You must provide --data_path if not using --use_public.")
        print(f"📂 Loading local dataset: {args.data_path}")
        raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    
    if args.use_public == "HuggingFaceH4/ultrafeedback_binarized":
        # special case: flatten chat messages
        def preprocess(example):
            return {
                "prompt": example["prompt"],        # user prompt
                "chosen": example["chosen"][1]["content"],        # assistant reply
                "rejected": example["rejected"][1]["content"]     # assistant reply
            }
    
    elif args.use_public == "Vezora/Code-Preference-Pairs":
        def preprocess(example):
            return {
                "prompt": example["input"],        # user prompt
                "chosen": example["accepted"],        # assistant reply
                "rejected": example["rejected"]     # assistant reply
            }
    
    elif args.use_public == "stanfordnlp/SHP":
        def preprocess(example):
            if example['labels'] == 1:
                return {
                    "prompt": example['history'],
                    "chosen": example['human_ref_B'],
                    "rejected": example['human_ref_A']
                }
            else:
                return {
                    "prompt": example['history'],
                    "chosen": example['human_ref_A'],
                    "rejected": example['human_ref_B']
                }
    
    elif args.use_public == "HuggingFaceH4/stack-exchange-preferences":
        def preprocess(example):
            question = example["question"]
            scored = sorted(
                [a for a in example["answers"] if "pm_score" in a],
                key=lambda x: x["pm_score"],
                reverse=True
            )
            chosen = scored[0]["text"]
            rejected = random.choice(scored[1:])["text"]

            return {
                "prompt": question,
                "chosen": chosen,
                "rejected": rejected
            }

    elif args.use_public == "openai/webgpt_comparisons":
        def preprocess(example):
            if example["score_0"] > example["score_1"]:
                return {
                    "prompt": example["question"]["full_text"],
                    "chosen": example["answer_0"],
                    "rejected": example["answer_1"],
                }
            else:
                return {
                    "prompt": example["question"]["full_text"],
                    "chosen": example["answer_1"],
                    "rejected": example["answer_0"],
                }


    else:
        # normal format (already strings)
        def preprocess(example):
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"]
        }

    dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
    random_seed = 42
    dataset = dataset.shuffle(seed=random_seed)

    print(dataset[0])

    if args.max_train_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
        print(f"📉 Truncated dataset to {len(dataset)} samples")

    dataset_size = len(dataset)
    model_repo = f"{args.hf_username}/dpo_model_{dataset_name}_{dataset_size}"
    output_dir = f"./{model_repo.replace('/', '_')}"

    print(f"📘 Dataset: {dataset_name}")
    print(f"✅ Loaded {dataset_size} examples")
    print(f"🚀 Model will be pushed to: {model_repo}")

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