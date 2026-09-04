"""CLI entrypoint for the project. Wraps scripts/train.py and scripts/evaluate.py.

Usage:
    python fine_tune_llama2.py train  (run training only)
    python fine_tune_llama2.py train --merge  (train andmerge LoRA adapter into base model after training)
    python fine_tune_llama2.py train --merge --push your-username/Llama-2-7b-chat-finetune (train, merge, and push to HF Hub)
    python fine_tune_llama2.py evaluate --model-path Llama-2-7b-chat-finetune-merged --prompt "..." (evaluate a trained model)
"""
import argparse  # library for parsing command-line arguments
import logging   # for printing info/error messages to the console

from scripts import evaluate as evaluate_mod
from scripts import train as train_mod

# Creates a logger instance for this module (fine_tune_llama2)

# Used to print logs with proper module name identification
logger = logging.getLogger(__name__)

# handles the train command.
def cmd_train(args):
    cfg = train_mod.load_config(args.config) # loads the configuration from the specified config.yaml file
    adapter_path = train_mod.train(cfg) # runs the training function with the loaded configuration
    logger.info("Adapter saved to %s", adapter_path) # prints a log message indicating where the adapter was saved

    if args.merge or args.push:
        merged_dir = train_mod.merge_and_save(cfg, adapter_path) # merge the adapter into the base model.
        logger.info("Merged model saved to %s", merged_dir)
        if args.push: # if the --push flag is set, push the merged model to the Hugging Face Hub.
            train_mod.push_to_hub(cfg, merged_dir, args.push)
            logger.info("Pushed to hub: %s", args.push)

# handles the evaluate command.
def cmd_evaluate(args):
    cfg = evaluate_mod.load_config(args.config) # loads the configuration from the specified config.yaml file
    prompts = args.prompts or [cfg["inference"]["default_prompt"]] # uses the provided prompts or the
    # default prompt from the config
    evaluate_mod.run_generation(args.model_path, prompts, cfg["inference"]["max_length"]) # runs the text 
    # generation evaluation on the specified model path with the given prompts and max length

# main cli argument parser function. Sets up the command-line interface for the project.
def main():
    parser = argparse.ArgumentParser(description="Llama 2 QLoRA fine-tuning project") # creates a new argument 
    #parser with a description of the project
    # adds a --config argument to specify the path to the config.yaml file (default is "config.yaml")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    # adds subparsers for the "train" and "evaluate" commands, each with their own specific arguments and help messages
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run QLoRA fine-tuning") # creates a subparser for the "train" 
    #command with a help message

    # adds a --merge flag to the "train" command to merge the LoRA adapter into the base model after training
    train_parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model after training")
    # adds a --push argument to the "train" command to specify a Hugging Face Hub repo id to push the merged model to
    train_parser.add_argument("--push", metavar="REPO_ID", default=None, help="Push merged model to this HF Hub repo id")
    train_parser.set_defaults(func=cmd_train) # sets the default function to call when the "train" command is used

    eval_parser = subparsers.add_parser("evaluate", help="Run text-generation checks against a trained model")
    eval_parser.add_argument("--model-path", required=True, help="Path or Hub id of the (merged) model to evaluate")
    eval_parser.add_argument("--prompt", action="append", dest="prompts", default=None, help="Prompt to test (repeatable)")
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args() # parses the command-line arguments and stores them in the args variable
    logging.basicConfig(level=logging.INFO) # sets up basic logging configuration to print INFO level logs to the console
    args.func(args) # calls the function associated with the chosen command (train or evaluate) and 
    #passes the parsed arguments to it


if __name__ == "__main__":
    main()
