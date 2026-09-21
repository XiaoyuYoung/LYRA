"""Fine-tune Qwen3-8B with per-head kappa/scale and calibrated t-vMF."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, set_seed
from safetensors.torch import load_file

try:
    from .data import CausalLMCollator, MultiSourceSFTDataset
    from .rope import training_rope_config
    from .modeling import (
        assert_strict_last_block,
        configure_trainable_parameters,
        V4Qwen3ForCausalLM, v4_config, load_checkpoint, head_parameter_values,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT,
        model_signature,
        register_backend,
        save_trainable_checkpoint,
        validate_qwen3_8b,
    )
    from .tvmf_attention import DEFAULT_KAPPA
except ImportError:  # Support direct execution from the source directory.
    from data import CausalLMCollator, MultiSourceSFTDataset
    from rope import training_rope_config
    from modeling import (
        assert_strict_last_block,
        configure_trainable_parameters,
        V4Qwen3ForCausalLM, v4_config, load_checkpoint, head_parameter_values,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT,
        model_signature,
        register_backend,
        save_trainable_checkpoint,
        validate_qwen3_8b,
    )
    from tvmf_attention import DEFAULT_KAPPA


def make_training_arguments(args: argparse.Namespace, *, dtype: torch.dtype, has_eval: bool) -> TrainingArguments:
    """Build TrainingArguments across Transformers 4.x and 5.x APIs."""
    periodic_saves = args.save_steps > 0
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "steps" if periodic_saves else "no",
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "length_column_name": "length",
        "seed": args.seed,
    }
    if periodic_saves:
        # Trainer checkpoints contain the full 8B model. The default is zero so
        # LYRA writes only the compact final checkpoint.
        kwargs["save_steps"] = args.save_steps
    supported = inspect.signature(TrainingArguments).parameters
    if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in supported:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    evaluation_value = "epoch" if has_eval else "no"
    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = evaluation_value
    else:
        kwargs["evaluation_strategy"] = evaluation_value
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = args.warmup_ratio
    else:
        kwargs["warmup_steps"] = args.warmup_ratio
    if "group_by_length" in supported:
        kwargs["group_by_length"] = args.group_by_length
    else:
        kwargs["train_sampling_strategy"] = "group_by_length" if args.group_by_length else "random"
    return TrainingArguments(**kwargs)


class SupervisedLogitsTrainer(Trainer):
    """Compute vocabulary logits only at positions with supervised targets."""

    def __init__(self, *args, supervised_logits_only: bool = True, head_learning_rate=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supervised_logits_only = supervised_logits_only
        self.head_learning_rate = head_learning_rate

    def get_decay_parameter_names(self, model):
        return [name for name in super().get_decay_parameter_names(model) if ".tvmf_parameters." not in name]

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        optimizer = super().create_optimizer()
        if self.head_learning_rate is not None:
            heads = {id(p) for n, p in self.model.named_parameters() if ".tvmf_parameters." in n}
            groups = []
            for group in optimizer.param_groups:
                normal = [p for p in group["params"] if id(p) not in heads]
                scalars = [p for p in group["params"] if id(p) in heads]
                if normal:
                    groups.append({**group, "params": normal})
                if scalars:
                    groups.append({**group, "params": scalars, "lr": self.head_learning_rate, "weight_decay": 0.0})
            optimizer.param_groups[:] = groups
        return optimizer

    def log(self, logs, *args, **kwargs):
        model = self.accelerator.unwrap_model(self.model)
        if getattr(model.config, "tvmf_version", None) == 4:
            logs = dict(logs)
            for layer, values in head_parameter_values(model).items():
                for name, numbers in values.items():
                    logs[f"tvmf/layer_{layer}/{name}_mean"] = sum(numbers) / len(numbers)
                    logs[f"tvmf/layer_{layer}/{name}_min"] = min(numbers)
                    logs[f"tvmf/layer_{layer}/{name}_max"] = max(numbers)
        return super().log(logs, *args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        if not self.supervised_logits_only:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Supervised-logits-only loss requires labels.")
        shift_labels = labels[..., 1:].contiguous()
        prediction_positions = shift_labels.ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
        if prediction_positions.numel() == 0:
            raise ValueError("Batch has no supervised causal-LM prediction targets.")

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs, logits_to_keep=prediction_positions)
        selected_labels = shift_labels.index_select(-1, prediction_positions).to(outputs.logits.device)
        logits = outputs.logits.float()
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            selected_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        denominator = selected_labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
        if torch.is_tensor(denominator):
            denominator = denominator.to(device=loss_sum.device, dtype=loss_sum.dtype)
        loss = loss_sum / denominator
        return (loss, outputs) if return_outputs else loss


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-checkpoint", help="Import compatible initialization weights and initialize LYRA scalars.")
    parser.add_argument("--resume-from-checkpoint", help="Resume a full LYRA Trainer checkpoint, including optimizer state.")
    parser.add_argument("--train-file", nargs="+", required=True, help="One or more normalized JSONL shards.")
    parser.add_argument("--eval-file", nargs="+")
    parser.add_argument("--source-weights", help="Comma-separated probability per training shard.")
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategy",
        choices=["last_attention", "last_block", "tvmf_blocks", "head_parameters", "full"],
        default="last_block",
    )
    parser.add_argument("--tvmf-layers", default="last", help="all, last, or comma-separated 0-based indices.")
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--head-scale", type=float, default=1.0, help="Initial per-query-head logit multiplier.")
    parser.add_argument("--learnable-kappa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learnable-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--similarity-mode", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--scale-mode", choices=["qk_norm", "sqrt_head_dim", "none"], default="qk_norm")
    parser.add_argument("--logit-scale", type=float)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--no-checkpoint-chunks", action="store_true")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--yarn-factor", type=float, help="Static YaRN for training: 2 for 64K or 4 for 128K; saved in checkpoints.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dataset-cache-dir", default="/tmp/lyra-datasets-cache")
    parser.add_argument("--overflow-strategy", choices=["error", "truncate_left"], default="error")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, help="Optional separate LR for kappa/scale (always zero weight decay).")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Optional full-model Trainer checkpoint interval; 0 keeps only the compact final checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-final-norm", action="store_true")
    parser.add_argument("--strict-last-layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--all-token-loss", action="store_true", help="Also compute loss on prompts/contexts.")
    parser.add_argument(
        "--supervised-logits-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only materialize supervised output logits to reduce 8B peak memory.",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="Only intended for tiny smoke tests.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.strict_last_layer and (args.strategy != "last_block" or args.tvmf_layers != "last"):
        raise ValueError(
            "Strict LYRA training requires --strategy last_block and --tvmf-layers last. "
            "Use --no-strict-last-layer only for an explicit ablation."
        )
    if not math.isfinite(args.kappa) or args.kappa < 0 or (args.learnable_kappa and args.kappa == 0):
        raise ValueError("--kappa must be positive when learnable, or non-negative when fixed")
    if not math.isfinite(args.head_scale) or args.head_scale <= 0:
        raise ValueError("--head-scale must be finite and positive")
    if args.head_learning_rate is not None and (not math.isfinite(args.head_learning_rate) or args.head_learning_rate <= 0):
        raise ValueError("--head-learning-rate must be finite and positive")
    if args.init_checkpoint and args.resume_from_checkpoint:
        raise ValueError("Choose either --init-checkpoint or --resume-from-checkpoint")
    if args.query_chunk_size <= 0 or args.max_length <= 0:
        raise ValueError("--query-chunk-size and --max-length must be positive")
    if args.save_steps < 0:
        raise ValueError("--save-steps must be non-negative")
    if args.yarn_factor is not None and (not math.isfinite(args.yarn_factor) or args.yarn_factor <= 1):
        raise ValueError("--yarn-factor must be finite and greater than 1")
    if args.yarn_factor is not None and args.max_length > 32768 * args.yarn_factor:
        raise ValueError("--max-length exceeds the selected YaRN capacity")
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    missing_data = [path for path in (args.train_file + (args.eval_file or [])) if not Path(path).is_file()]
    if missing_data:
        raise FileNotFoundError(f"Data files do not exist: {missing_data}")
    output = Path(args.output_dir).expanduser().resolve()
    for source in [args.model, args.init_checkpoint, args.resume_from_checkpoint]:
        if source and output == Path(source).expanduser().resolve():
            raise ValueError("Use a separate output directory from the source checkpoint")


def initialize_model(args, dtype, *, validate_architecture=False):
    """Separate importing v3 weights from resuming trained v4 parameters."""
    if args.resume_from_checkpoint:
        path = Path(args.resume_from_checkpoint)
        if not (path / "trainer_state.json").is_file() or (path / "trainable.safetensors").is_file():
            raise ValueError("Resume requires a full v4 Trainer checkpoint")
        if validate_architecture:
            validate_qwen3_8b(AutoConfig.from_pretrained(path, local_files_only=True))
        resume_config = training_rope_config(AutoConfig.from_pretrained(path, local_files_only=True),
                                             args.max_length, args.yarn_factor, resume=True)
        model = load_checkpoint(path, dtype=dtype, config=resume_config)
        if args.strict_last_layer and model.config.tvmf_layer_indices != [model.config.num_hidden_layers - 1]:
            raise ValueError("Resumed checkpoint is not a strict last-layer experiment")
        args.tvmf_layers = ",".join(map(str, model.config.tvmf_layer_indices))
        base = model.config.tvmf_base_model
        return model, base, path
    source = Path(args.init_checkpoint or args.model).expanduser().resolve()
    compact = (source / "trainable.safetensors").is_file()
    weights = Path(args.model).expanduser().resolve() if compact else source
    config = AutoConfig.from_pretrained(weights, local_files_only=True)
    if validate_architecture:
        validate_qwen3_8b(config)
    if getattr(config, "tvmf_version", None) == 4:
        raise ValueError("Use --resume-from-checkpoint for v4; --init-checkpoint imports v3 or base weights")
    if args.strict_last_layer and getattr(config, "tvmf_layer_indices", [config.num_hidden_layers - 1]) != [config.num_hidden_layers - 1]:
        raise ValueError("Source checkpoint has non-final t-vMF layers; select an explicit ablation")
    config = v4_config(config, kappa=args.kappa, head_scale=args.head_scale,
                       learnable_kappa=args.learnable_kappa, learnable_scale=args.learnable_scale,
                       similarity_mode=args.similarity_mode, scale_mode=args.scale_mode,
                       logit_scale=args.logit_scale, layers=args.tvmf_layers,
                       query_chunk_size=args.query_chunk_size, checkpoint_chunks=not args.no_checkpoint_chunks)
    config = training_rope_config(config, args.max_length, args.yarn_factor)
    # Do not persist local filesystem paths in checkpoints. Compact checkpoints
    # are loaded with an explicit ``--base-model`` argument.
    config.tvmf_base_model = None
    config.tvmf_benchmark_contamination = (
        True if getattr(config, "tvmf_benchmark_contamination", None) is True
        or args.init_checkpoint or any("longbench" in path.lower() for path in args.train_file) else None
    )
    model, info = V4Qwen3ForCausalLM.from_pretrained(
        weights, config=config, dtype=dtype, local_files_only=True, output_loading_info=True,
    )
    bad = [n for n in info.get("missing_keys", []) if ".tvmf_parameters." not in n]
    if bad or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Initialization weights mismatch: {info}")
    if compact:
        metadata = json.loads((source / "tvmf_v3_config.json").read_text())
        if metadata.get("format_version") != 3:
            raise ValueError("Expected compact v3 checkpoint")
        state = load_file(str(source / "trainable.safetensors"))
        if not state or model.load_state_dict(state, strict=False).unexpected_keys:
            raise ValueError("Invalid compact v3 checkpoint")
        model.config.tvmf_inherited_state_keys = sorted(state)
    return model, str(weights), source


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA device is visible. Request a GPU node or pass --allow-cpu for a tiny test.")
    set_seed(args.seed)
    register_backend()

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model, base_model, tokenizer_path = initialize_model(args, dtype, validate_architecture=True)
    tokenizer_path = tokenizer_path if (tokenizer_path / "tokenizer_config.json").is_file() else Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    selected_layers = model.config.tvmf_layer_indices
    parameter_stats = configure_trainable_parameters(
        model,
        args.strategy,
        tvmf_layers=args.tvmf_layers,
        train_final_norm=args.train_final_norm,
    )
    if args.strict_last_layer:
        assert_strict_last_block(model, allow_final_norm=args.train_final_norm)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    source_weights = None
    if args.source_weights:
        source_weights = [float(value) for value in args.source_weights.split(",")]
    dataset_kwargs = {
        "max_length": args.max_length,
        "response_only": not args.all_token_loss,
        "cache_dir": args.dataset_cache_dir,
        "seed": args.seed,
        "overflow_strategy": args.overflow_strategy,
    }
    train_dataset = MultiSourceSFTDataset(
        args.train_file,
        tokenizer,
        source_weights=source_weights,
        samples_per_epoch=args.samples_per_epoch,
        max_samples=args.max_samples,
        **dataset_kwargs,
    )
    eval_dataset = None
    if args.eval_file:
        eval_dataset = MultiSourceSFTDataset(args.eval_file, tokenizer, **dataset_kwargs)

    training_args = make_training_arguments(args, dtype=dtype, has_eval=eval_dataset is not None)
    trainer = SupervisedLogitsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        supervised_logits_only=args.supervised_logits_only,
        head_learning_rate=args.head_learning_rate,
    )

    safe_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {
            "model", "init_checkpoint", "resume_from_checkpoint", "train_file",
            "eval_file", "output_dir", "dataset_cache_dir",
        }
    }
    summary = {
        "format_version": 4,
        **safe_args,
        "base_model": None,
        "train_files": [Path(path).name for path in args.train_file],
        "eval_files": [Path(path).name for path in (args.eval_file or [])],
        "rope_parameters": getattr(model.config, "rope_parameters", None) or getattr(model.config, "rope_scaling", None),
        **parameter_stats,
        "model_signature": model_signature(model.config),
        "selected_tvmf_layers": list(selected_layers),
        "dtype": str(dtype),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    if trainer.is_world_process_zero():
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        summary["train_metrics"] = result.metrics
        save_trainable_checkpoint(model, tokenizer, args.output_dir, summary)
        print(f"Saved compact LYRA checkpoint to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
