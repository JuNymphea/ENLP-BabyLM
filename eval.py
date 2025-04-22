import argparse
import json
import os
from math import exp
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
)
from quinine import Quinfig


def main(quinfig, config_file, model_name, checkpoint_number):
    checkpoint_path = f"checkpoints/finetune/gpt2_small/{model_name}/checkpoint-{checkpoint_number}"

    tokenizer = AutoTokenizer.from_pretrained(quinfig.data.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(quinfig.dataset.id)
    block_size = 512

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=block_size)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated_examples["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000)
    lm_dataset.set_format(type="torch")

    model_type = quinfig.general.model_type
    if "gpt2" in model_type:
        model_cls = AutoModelForCausalLM
    elif "bert" in model_type:
        model_cls = AutoModelForMaskedLM
    else:
        raise ValueError("Unsupported model type")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = model_cls.from_pretrained(checkpoint_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    run_name = f"{quinfig.general.nickname}_eval"

    if "test" not in lm_dataset:
        print("No test split found in dataset. Exiting.")
        return

    print("Running evaluation on test set...")
    test_metrics = trainer.evaluate(eval_dataset=lm_dataset["test"])

    if "eval_loss" in test_metrics:
        test_ppl = exp(test_metrics["eval_loss"])
        print(f"Test Perplexity: {test_ppl:.2f}")
        test_metrics["perplexity"] = test_ppl

    # ✅ 添加 checkpoint number 到结果
    test_metrics["checkpoint_number"] = checkpoint_number

    output_file = f"eval/{run_name}_test_metrics.json"

    # Load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Append new result
    all_results.append(test_metrics)

    # Save back to file
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--checkpoint-number", type=int, required=True)
    args = parser.parse_args()

    quinfig = Quinfig(args.config_file)
    config_filename = os.path.splitext(os.path.basename(args.config_file))[0]

    main(quinfig, config_filename, args.model_name, args.checkpoint_number)
