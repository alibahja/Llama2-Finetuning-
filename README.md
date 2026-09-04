# Llama 2 Fine-tuning (QLoRA)

Fine-tunes `NousResearch/Llama-2-7b-chat-hf` on the `mlabonne/guanaco-llama2-1k` instruction
dataset using QLoRA (4-bit NF4 quantization + LoRA), then optionally merges the adapter into
the base model and pushes it to the Hugging Face Hub. Restructured from a single Colab
notebook into a CLI-runnable project.

## Project structure

```
llama2-finetuning/
├── README.md
├── requirements.txt
├── config.yaml              # All hyperparameters (model, LoRA, bnb, training, SFT)
├── fine_tune_llama2.ipynb   # Notebook version, wraps scripts/
├── fine_tune_llama2.py      # CLI entrypoint (train / evaluate)
├── scripts/
│   ├── train.py             # QLoRA training, merge, hub push
│   └── evaluate.py          # Text-generation checks
├── data/
│   └── dataset_info.json    # Dataset metadata / prompt template
├── results/
│   ├── checkpoints/
│   └── logs/                # TensorBoard logs
└── .gitignore
```

## Requirements

- Python 3.13 (also works on 3.10+)
- An NVIDIA GPU with at least ~15 GB VRAM (e.g. a Colab T4) for 4-bit QLoRA on the 7B model
- A Hugging Face account/token only if you plan to push the merged model to the Hub

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you'll push to the Hub, set a token instead of using the interactive CLI login:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

## Usage

Train:

```bash
python fine_tune_llama2.py train
```

Train, then merge the LoRA adapter into the base model:

```bash
python fine_tune_llama2.py train --merge
```

Train, merge, and push to the Hub:

```bash
python fine_tune_llama2.py train --merge --push your-username/Llama-2-7b-chat-finetune
```

Evaluate a trained model with the Llama 2 chat prompt template:

```bash
python fine_tune_llama2.py evaluate \
  --model-path Llama-2-7b-chat-finetune-merged \
  --prompt "What is a large language model?"
```

Each script under `scripts/` is also runnable standalone, e.g. `python scripts/train.py --config config.yaml --merge`.

### Notebook

`fine_tune_llama2.ipynb` walks through the same steps interactively (data loading,
training, TensorBoard, generation check, merge, Hub push) by calling into `scripts/train.py`
and `scripts/evaluate.py`, so notebook and CLI stay in sync.

## Configuration

All hyperparameters live in `config.yaml`: base model, dataset, LoRA (`r`, `alpha`, `dropout`),
4-bit quantization (`bnb`), `TrainingArguments`-equivalent fields, and SFT settings
(`max_seq_length`, `packing`). Edit this file rather than the scripts to change behavior.

## Notes on Python 3.13 compatibility

The original notebook pinned 2023-era versions (`transformers==4.31.0`, `trl==0.4.7`,
`peft==0.4.0`, `accelerate==0.21.0`) that predate Python 3.13 wheels, and used an older
`SFTTrainer` API. `requirements.txt` here uses modern floor versions with cp313 wheels, and
`scripts/train.py` uses the current `trl` API: hyperparameters that used to live directly on
`SFTTrainer` (`max_seq_length`, `packing`, `dataset_text_field`) now live on `SFTConfig`, and
the `tokenizer=` kwarg was renamed to `processing_class=`.

## License

Base model and dataset each carry their own licenses on the Hugging Face Hub — check
[`NousResearch/Llama-2-7b-chat-hf`](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) and
[`mlabonne/guanaco-llama2-1k`](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
before redistributing fine-tuned weights.
