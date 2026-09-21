# LYRA



This repository is a PyTorch implementation of LYRA proposed in *The Sirens’ Song: When Proximal Background Context Overshadows Distant Evidence* 

Long-context LLMs focus on retrieving distant evidence from extensive context, yet existing work has largely focused on overcoming distance alone.
In this work, we identify the Proximity Trap, insufficient attention to distant evidence often arises less from distance itself than from cumulative competition with abundant, task-irrelevant proximal background. 
To address the Proximity Trap, we introduce LYRA (\textbf{L}ong-context heav\textbf{Y}-tailed \textbf{R}elevance \textbf{A}lignment), a T-distributed directional matching mechanism that reshapes the context retrieval distribution to redistribute attention mass toward task-relevant evidence, while preserving the relative positional information encoded.
Extensive experiments on LongBench-v2, RULER, and LongBench demonstrate consistent improvements across context lengths and task categories. We further introduce ProxBench, a multi-level fine-grained benchmark for evaluating distant evidence utilization under increasing proximal background interference.




## Installation

Python 3.10 or newer is required. Create an isolated environment and install the
package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,eval]'
```

LYRA loads models from local files by default. One way to obtain Qwen3-8B with the
current Hugging Face CLI is:

```bash
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

The `models/` directory is ignored by Git. Review and comply with the model's
license and terms before use.

## Training data

Training inputs must remain outside version control. LYRA accepts local JSON or
JSONL files, optionally gzip-compressed. Supported records include either:

- a `messages` or `conversations` array with user and assistant turns; or
- an instruction/question plus an `answer`, `output`, or `response` field.

The final assistant turn is used as the supervised target by default. Inputs that
exceed `--max-length` raise an error unless truncation is explicitly enabled.

Use only data that you are permitted to process. Keep evaluation benchmarks out
of training data, and document dataset provenance in your own experiment records.

## Training

The included launcher requires paths through environment variables and contains
no machine-specific defaults:

```bash
MODEL_DIR=./models/Qwen3-8B \
TRAIN_FILE=/path/to/train.jsonl \
OUTPUT_DIR=./outputs/lyra-qwen3-8b \
bash scripts/train_2gpu.sh
```

Equivalent direct invocation:

```bash
torchrun --standalone --nproc_per_node=2 -m lyra.train \
  --model ./models/Qwen3-8B \
  --train-file /path/to/train.jsonl \
  --output-dir ./outputs/lyra-qwen3-8b \
  --max-length 16384 \
  --strategy last_block \
  --tvmf-layers last \
  --gradient-checkpointing
```

The final compact checkpoint stores trainable tensors and tokenizer files. Local
source paths are not embedded in its public metadata. Pass the base model explicitly
when loading a compact checkpoint on another machine.

## Inference

```bash
python -m lyra.generate \
  --checkpoint ./outputs/lyra-qwen3-8b \
  --base-model ./models/Qwen3-8B \
  --prompt "Summarize the role of long-context attention."
```

## Evaluation

Evaluation data is not included. Obtain LongBench or LongBench-v2 from their
official repositories and keep the files under an ignored directory such as
`external_data/`.

```bash
CHECKPOINT=./outputs/lyra-qwen3-8b \
BASE_MODEL=./models/Qwen3-8B \
LONGBENCH_V2=./external_data/LongBench-v2/data.json \
bash scripts/evaluate_longbench_v2.sh

python -m lyra.score_longbench --results-dir ./outputs/evaluation
```

Relevant upstream projects:

- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
- [LongBench](https://github.com/THUDM/LongBench)
- [LongBench-v2](https://github.com/THUDM/LongBench-v2)


