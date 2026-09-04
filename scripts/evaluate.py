"""Run the Llama 2 prompt-template text-generation check from notebook Step 6."""
import argparse # library for parsing command-line arguments
import logging as pylogging # for printing info/error messages to the console

import yaml # library for parsing YAML configuration files
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, pipeline # libraries from Hugging Face 
#Transformers for model loading, tokenization, and text generation

logger = pylogging.getLogger(__name__) # creates a logger instance for this module (evaluate.py) to print 
#logs with proper module name identification

# loads the configuration from the specified config.yaml file and returns it as a dictionary.
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# load a fine-tuned Llama 2 model and run text generation on the provided prompts, returning the generated texts.
def run_generation(model_path: str, prompts: list[str], max_length: int) -> list[str]:
    logging.set_verbosity(logging.CRITICAL)  # silence generation warnings, matches notebook

    model = AutoModelForCausalLM.from_pretrained(model_path) # loads the fine-tuned model from the specified path
    tokenizer = AutoTokenizer.from_pretrained(model_path) # loads the tokenizer associated with the fine-tuned model
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length) # creates a 
    #text-generation pipeline using the loaded model and tokenizer, with the specified max length

    results = []
    for prompt in prompts:
        output = pipe(f"<s>[INST] {prompt} [/INST]") #Format prompt: Wraps in Llama 2's chat template: <s>[INST] {prompt} [/INST]
        # <s> is the start of the sequesnce token, [INST] and [/INST] are start and end instruction tokens
        text = output[0]["generated_text"]
        results.append(text) # add to the result list 
        print(text) #show the output to the user with a separator line for clarity
        print("-" * 80)
    return results # return all generated texts as a list


def main():
    # sets up the command-line interface for the evaluation script.
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama 2 model") # creates a new argument 
    #parser with a description of the script
    # adds a --config argument to specify the path to the config.yaml file (default is "config.yaml")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    # adds a --model-path argument to specify the path or Hub id of the (merged) model to evaluate
    parser.add_argument("--model-path", required=True, help="Path or Hub id of the (merged) model to evaluate")
    # adds a --prompt argument to specify the prompt to test (repeatable)
    parser.add_argument("--prompt", action="append", dest="prompts", default=None, help="Prompt to test (repeatable)")
    args = parser.parse_args() #parses the command-line arguments and stores them in the args variable

    pylogging.basicConfig(level=pylogging.INFO) # sets up basic logging configuration to print INFO level logs to the console
    cfg = load_config(args.config) # loads the config.yaml file and stores it in the cfg variable
    prompts = args.prompts or [cfg["inference"]["default_prompt"]] # uses the provided prompts or
    #the default prompt from the config

    run_generation(args.model_path, prompts, cfg["inference"]["max_length"]) # runs the text generation 
    #evaluation on the specified model path with the given prompts and max length


if __name__ == "__main__":
    main()
