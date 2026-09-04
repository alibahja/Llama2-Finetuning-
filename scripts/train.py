"""QLoRA fine-tuning of Llama 2 with a modern (Python 3.13-compatible) stack.

Mirrors the original Colab notebook's Steps 1-8, adapted to the current
transformers/trl/peft API:
  - trl>=0.9 moved SFT hyperparameters (max_seq_length, packing,
    dataset_text_field) off SFTTrainer and onto SFTConfig.
  - SFTTrainer's `tokenizer=` kwarg was renamed to `processing_class=`.
"""
import argparse
import gc
import logging as pylogging
import os

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logger = pylogging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def maybe_hf_login() -> None:
    """Non-interactive HF Hub auth. Set HF_TOKEN in the environment (or a
    .env file) instead of running `huggingface-cli login` interactively."""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        logger.info("HF_TOKEN not set; assuming a cached `huggingface-cli login` or a public model/dataset.")


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    bnb_cfg = cfg["bnb"]
    compute_dtype = getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"])
    if compute_dtype == torch.float16 and bnb_cfg["use_4bit"] and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("GPU supports bfloat16 - consider setting bnb.bnb_4bit_compute_dtype: bfloat16")
    return BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["use_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_cfg["use_nested_quant"],
    )


def load_model_and_tokenizer(cfg: dict):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=build_bnb_config(cfg),
        device_map=cfg["device_map"],
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # avoids overflow issues with fp16 training
    return model, tokenizer


def build_peft_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]
    return LoraConfig(
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        r=lora_cfg["r"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    s = cfg["sft"]
    return SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        #logging_dir=t["logging_dir"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        max_grad_norm=t["max_grad_norm"],
        max_steps=t["max_steps"],
        warmup_ratio=t["warmup_ratio"],
        group_by_length=t["group_by_length"],
        lr_scheduler_type=t["lr_scheduler_type"],
        report_to="tensorboard",
        dataset_text_field=cfg["dataset"]["text_field"],
        max_seq_length=s["max_seq_length"],
        packing=s["packing"],
    )


def train(cfg: dict) -> str:
    """Runs QLoRA SFT and saves the LoRA adapter. Returns the adapter path."""
    maybe_hf_login()
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    model, tokenizer = load_model_and_tokenizer(cfg)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=build_peft_config(cfg),
        processing_class=tokenizer,
        args=build_sft_config(cfg),
    )
    trainer.train()

    adapter_path = cfg["model"]["new_model_name"]
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_path


def merge_and_save(cfg: dict, adapter_path: str, merged_dir: str = None) -> str:
    """Reloads the base model in fp16 and merges the LoRA adapter into it."""
    merged_dir = merged_dir or f"{cfg['model']['new_model_name']}-merged"
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=cfg["device_map"],
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def push_to_hub(cfg: dict, merged_dir: str, repo_id: str) -> None:
    maybe_hf_login()
    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model.push_to_hub(repo_id, check_pr=True)
    tokenizer.push_to_hub(repo_id, check_pr=True)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Llama 2")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model after training")
    parser.add_argument("--push", metavar="REPO_ID", default=None, help="Push merged model to this HF Hub repo id")
    args = parser.parse_args()

    pylogging.basicConfig(level=pylogging.INFO)
    cfg = load_config(args.config)

    adapter_path = train(cfg)
    logger.info("Adapter saved to %s", adapter_path)

    if args.merge or args.push:
        merged_dir = merge_and_save(cfg, adapter_path)
        logger.info("Merged model saved to %s", merged_dir)
        if args.push:
            push_to_hub(cfg, merged_dir, args.push)
            logger.info("Pushed to hub: %s", args.push)


if __name__ == "__main__":
    main()
