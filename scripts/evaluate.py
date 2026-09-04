"""Run the Llama 2 prompt-template text-generation check from notebook Step 6."""
import argparse
import logging as pylogging

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, pipeline

logger = pylogging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_generation(model_path: str, prompts: list[str], max_length: int) -> list[str]:
    logging.set_verbosity(logging.CRITICAL)  # silence generation warnings, matches notebook

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)

    results = []
    for prompt in prompts:
        output = pipe(f"<s>[INST] {prompt} [/INST]")
        text = output[0]["generated_text"]
        results.append(text)
        print(text)
        print("-" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama 2 model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model-path", required=True, help="Path or Hub id of the (merged) model to evaluate")
    parser.add_argument("--prompt", action="append", dest="prompts", default=None, help="Prompt to test (repeatable)")
    args = parser.parse_args()

    pylogging.basicConfig(level=pylogging.INFO)
    cfg = load_config(args.config)
    prompts = args.prompts or [cfg["inference"]["default_prompt"]]

    run_generation(args.model_path, prompts, cfg["inference"]["max_length"])


if __name__ == "__main__":
    main()
