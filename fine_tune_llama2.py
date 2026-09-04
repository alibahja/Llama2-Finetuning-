"""CLI entrypoint for the project. Wraps scripts/train.py and scripts/evaluate.py.

Usage:
    python fine_tune_llama2.py train
    python fine_tune_llama2.py train --merge
    python fine_tune_llama2.py train --merge --push your-username/Llama-2-7b-chat-finetune
    python fine_tune_llama2.py evaluate --model-path Llama-2-7b-chat-finetune-merged --prompt "..."
"""
import argparse
import logging

from scripts import evaluate as evaluate_mod
from scripts import train as train_mod

logger = logging.getLogger(__name__)


def cmd_train(args):
    cfg = train_mod.load_config(args.config)
    adapter_path = train_mod.train(cfg)
    logger.info("Adapter saved to %s", adapter_path)

    if args.merge or args.push:
        merged_dir = train_mod.merge_and_save(cfg, adapter_path)
        logger.info("Merged model saved to %s", merged_dir)
        if args.push:
            train_mod.push_to_hub(cfg, merged_dir, args.push)
            logger.info("Pushed to hub: %s", args.push)


def cmd_evaluate(args):
    cfg = evaluate_mod.load_config(args.config)
    prompts = args.prompts or [cfg["inference"]["default_prompt"]]
    evaluate_mod.run_generation(args.model_path, prompts, cfg["inference"]["max_length"])


def main():
    parser = argparse.ArgumentParser(description="Llama 2 QLoRA fine-tuning project")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run QLoRA fine-tuning")
    train_parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model after training")
    train_parser.add_argument("--push", metavar="REPO_ID", default=None, help="Push merged model to this HF Hub repo id")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("evaluate", help="Run text-generation checks against a trained model")
    eval_parser.add_argument("--model-path", required=True, help="Path or Hub id of the (merged) model to evaluate")
    eval_parser.add_argument("--prompt", action="append", dest="prompts", default=None, help="Prompt to test (repeatable)")
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
