"""QLoRA fine-tuning of Llama 2 with a modern (Python 3.13-compatible) stack.
"""
import argparse # Command-line argument parser (for running from terminal)
import gc #Garbage collection (to free up memory after training)
import logging as pylogging # print progress and debug informations
import os # Environment variables (for HF_TOKEN)

import torch #PyTorch library (for model training and inference)
import yaml # Parse config.yaml
from datasets import load_dataset #loads data from hugging face datasets 
from huggingface_hub import login #Authenticate to hugging face
from peft import LoraConfig, PeftModel #Parameter-Efficient Fine-Tuning (PEFT) library for LoRA
# LoraConfig: Configuration for LoRA (Low-Rank Adaptation) fine-tuning and PeftModel: Wrapper for models with LoRA adapters
from transformers import ( #Huggin Face models and tokenizers 
    AutoModelForCausalLM, #Generic causual learning model (Llama)
    AutoTokenizer, # Tokenizer for the model (Llama)
    BitsAndBytesConfig, # 4-bit quantization configuration for memory-efficient training and inference
)
from trl import SFTConfig, SFTTrainer # Transformer Reinforcement Learning (TRL) library for supervised fine-tuning (SFT)
# SFTConfig: Configuration for supervised fine-tuning and SFTTrainer: Trainer class for supervised fine-tuning

#Creates a logger instance for this module. __name__ is the module name (train). Used for printing logs with proper formatting.
logger = pylogging.getLogger(__name__)

# loads the YAML configuration file and returns its python dictionary
def load_config(config_path: str) -> dict: #input is the path to the config.yaml 
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

#Authenticate to Hugging Face Hub if the token exists in the environment variable 
# HF_TOKEN. If not, it assumes that the user has already logged in interactively or is using a public model/dataset.
#Why needed: some datasets/models require authentication to acess and downloads.
def maybe_hf_login() -> None:
    """Non-interactive HF Hub auth. Set HF_TOKEN in the environment (or a
    .env file) instead of running `huggingface-cli login` interactively."""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        logger.info("HF_TOKEN not set; assuming a cached `huggingface-cli login` or a public model/dataset.")

# creates a 4-bit quantization configuration for memory-efficient training (QLoRA)
def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    bnb_cfg = cfg["bnb"] #extracts the bnb section from the config dictionary
    #gets the compute dtype (float16,bfloat16) for 4-bit quantization from the config and converts it to a torch dtype. 
    compute_dtype = getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"])
    #If using float16 on a modern GPU (compute capability >= 8), suggest using bfloat16 (more stable)
    if compute_dtype == torch.float16 and bnb_cfg["use_4bit"] and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("GPU supports bfloat16 - consider setting bnb.bnb_4bit_compute_dtype: bfloat16")
    # returns a BitsAndBytesConfig object with the specified quantization settings from the config dictionary.
    return BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["use_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_cfg["use_nested_quant"],
    )

#loads the base model and tokenizer from Hugging Face Hub, applies 4-bit quantization if specified, and returns them.
def load_model_and_tokenizer(cfg: dict):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], # name of the model to load 
        quantization_config=build_bnb_config(cfg), # 4-bit quanntization configuration
        device_map=cfg["device_map"], # specifies which device (CPU/GPU) to load the model ontos
    )
    model.config.use_cache = False # disables caching for memory efficiency
    model.config.pretraining_tp = 1 # sets the number of tensor parallelism groups to 1 (no parallelism)
    # loads the tokenizer for the model, sets the padding token to the end-of-sequence
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # avoids overflow issues with fp16 training
    return model, tokenizer 

# Configures the LoRA settings 
def build_peft_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]
    #returns a LoraConfig object with the specified LoRA settings from the config dictionary.
    return LoraConfig(
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        r=lora_cfg["r"],
        bias="none",
        task_type="CAUSAL_LM",
    )

# Configures the supervised fine-tuning (SFT) settings
def build_sft_config(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    s = cfg["sft"]
    # returns a SFTConfig object with the specified training and SFT settings from the config dictionary.
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
        #warmup_ratio=t["warmup_ratio"],
        #group_by_length=t["group_by_length"],
        #lr_scheduler_type=t["lr_scheduler_type"],
        report_to="tensorboard",
        dataset_text_field=cfg["dataset"]["text_field"],
        max_length=s["max_seq_length"],
        packing=s["packing"],
    )

# Main training loop
def train(cfg: dict) -> str:
    """Runs QLoRA SFT and saves the LoRA adapter. Returns the adapter path."""
    maybe_hf_login() # authenticate to Hugging Face Hub if HF_TOKEN is set in the environment 
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"]) # load dataset from Hugging Face
    model, tokenizer = load_model_and_tokenizer(cfg) # load the base model and tokenizer from Hugging Face Hub
    # creates an SFTTrainer instance with the model, dataset, LoRA configuration, tokenizer, and training settings.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=build_peft_config(cfg),
        processing_class=tokenizer,
        args=build_sft_config(cfg),
    )
    # run the training (actual fine tuning of the model with LoRA adapters)
    trainer.train()
    # save the adapter (LoRA weights, not the full model) this is small.
    adapter_path = cfg["model"]["new_model_name"]
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path) # saves the tokenizer to the same directory (needed for inference)
    # Free memory - delete model/trainer and clear CUDA cache
    del model, trainer
    gc.collect() # run garbage collection to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # clear the GPU memory cache

    return adapter_path

# merges the LoRA adapter into the base model and saves it to disk. Returns the path to the merged model.
def merge_and_save(cfg: dict, adapter_path: str, merged_dir: str = None) -> str:
    """Reloads the base model in fp16 and merges the LoRA adapter into it."""
    merged_dir = merged_dir or f"{cfg['model']['new_model_name']}-merged" # gets the path to save the merged model.
    base_model = AutoModelForCausalLM.from_pretrained( # reload the base model in fp16 (not 4-bit)
        cfg["model"]["name"],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=cfg["device_map"],
    )
    model = PeftModel.from_pretrained(base_model, adapter_path) # loads the LoRA adapter into the base model 
    # (PeftModel is a wrapper that allows merging)
    model = model.merge_and_unload() # merges the LoRA adapter weights into the base model and unloads the adapter

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True) # reloads the tokenizer for 
    #the base model
    tokenizer.pad_token = tokenizer.eos_token # sets the padding token to the end-of-sequence token (needed for inference)
    tokenizer.padding_side = "right" # sets the padding side to right (needed for inference)
    # saves the merged model and tokenizer to disk (merged_dir)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir

# Upload merged model to Hugging Face Hub.
def push_to_hub(cfg: dict, merged_dir: str, repo_id: str) -> None:
    maybe_hf_login() # requires authentication 
    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float16) # reloads the merged model in fp16
    tokenizer = AutoTokenizer.from_pretrained(merged_dir) # reloads the tokenizer for the merged model
    model.push_to_hub(repo_id, check_pr=True) #push the merged model to the specified Hugging Face Hub repository (repo_id) 
    #and checks for pull requests
    tokenizer.push_to_hub(repo_id, check_pr=True) # pushes the tokenizer to the same repository 
    #(repo_id) and checks for pull requests

# Main function to parse command-line arguments and run the training, merging, and pushing steps.
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Llama 2") # creates an argument parser for 
    #command-line arguments

    # --config: adds command-line arguments for the config file path 
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml") 
    # --merge: adds a command-line argument to merge the LoRA adapter into the base model after training
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model after training")
    # --push: adds a command-line argument to push the merged model to the Hugging Face Hub
    parser.add_argument("--push", metavar="REPO_ID", default=None, help="Push merged model to this HF Hub repo id")
    # parses the command-line arguments and stores them in the args variable
    args = parser.parse_args()

    # sets up basic logging configuration to print INFO-level messages to the console.
    pylogging.basicConfig(level=pylogging.INFO)
    # loads the configuration from the specified config.yaml file and stores it in the cfg variable.
    cfg = load_config(args.config)
    # runs the training function with the loaded configuration and saves the path to the LoRA adapter.
    adapter_path = train(cfg)
    logger.info("Adapter saved to %s", adapter_path)
    # If the --merge or --push flags are set, merge the LoRA adapter into the base model and 
    # optionally push it to the Hugging Face Hub.
    if args.merge or args.push:
        merged_dir = merge_and_save(cfg, adapter_path)
        logger.info("Merged model saved to %s", merged_dir)
        if args.push:
            push_to_hub(cfg, merged_dir, args.push)
            logger.info("Pushed to hub: %s", args.push)


if __name__ == "__main__":
    main()
