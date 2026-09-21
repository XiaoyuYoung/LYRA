"""Evaluate LYRA full/compact checkpoints on LongBench, optionally using YaRN."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

if __package__:
    from . import benchmark as bench
    from .rope import yarn_config
    from .modeling import CONFIG_FILENAME, DEFAULT_OUTPUT, attention_recipe, load_checkpoint, validate_qwen3_8b
else:
    import benchmark as bench
    from rope import yarn_config
    from modeling import CONFIG_FILENAME, DEFAULT_OUTPUT, attention_recipe, load_checkpoint, validate_qwen3_8b


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, help="Optional compact-checkpoint base override; metadata is used by default.")
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--yarn-factor", type=float, help="Optional runtime YaRN (2 for 64K, 4 for 128K).")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--benchmark", choices=("longbench", "longbench-v2", "both"), default="longbench-v2")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--longbench", type=Path, default=bench.DEFAULT_LONGBENCH)
    parser.add_argument("--longbench-v2", type=Path, default=bench.DEFAULT_LONGBENCH_V2)
    parser.add_argument("--v2-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.query_chunk_size <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("query-chunk-size and max-samples must be positive")
    if args.yarn_factor is not None:
        if not math.isfinite(args.yarn_factor) or args.yarn_factor <= 1:
            parser.error("yarn-factor must be finite and greater than 1")
        if args.max_context_tokens > 32768 * args.yarn_factor:
            parser.error("Context limit exceeds the configured YaRN capacity")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    suffix = f"yarn{args.yarn_factor:g}-" if args.yarn_factor is not None else ""
    args.output_dir = (args.output_dir or Path(DEFAULT_OUTPUT) / f"results-{suffix}{args.max_context_tokens}").expanduser().resolve()
    if args.output_dir == args.checkpoint or args.output_dir in args.checkpoint.parents:
        parser.error("Evaluation output must be separate from the checkpoint directory")
    try:
        args.checkpoint_shard = int(os.environ.get("RANK", "0"))
        args.num_checkpoint_shards = int(os.environ.get("WORLD_SIZE", "1"))
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        parser.error(f"Invalid distributed rank environment: {exc}")
    if args.num_checkpoint_shards > 1:
        args.device = f"cuda:{args.local_rank}"
    return args


def model_plan(args):
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    validate_qwen3_8b(config)
    is_v4 = getattr(config, "tvmf_version", None) == 4
    if not is_v4 and getattr(config, "tvmf_layer_indices", None):
        raise ValueError("Use v3's evaluator for v3 checkpoints, or train v4 with --init-checkpoint")
    if args.yarn_factor is not None:
        config = yarn_config(config, args.yarn_factor)
    if args.max_context_tokens > config.max_position_embeddings:
        raise ValueError("Context budget exceeds model capacity; set an appropriate --yarn-factor")
    if is_v4:
        config.tvmf_query_chunk_size = args.query_chunk_size
        config.tvmf_checkpoint_chunks = False
    return config, is_v4


def load_model(args, config, is_v4, dtype):
    if is_v4:
        model = load_checkpoint(args.checkpoint, base_model=args.base_model, config=config, dtype=dtype)
    else:
        model = Qwen3ForCausalLM.from_pretrained(args.checkpoint, config=config, dtype=dtype,
                                                local_files_only=True, attn_implementation="sdpa")
    tokenizer_path = args.checkpoint
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        metadata = json.loads((args.checkpoint / CONFIG_FILENAME).read_text())
        tokenizer_path = args.base_model or metadata["base_model"]
        if not tokenizer_path:
            raise ValueError("Compact checkpoints without tokenizer files require --base-model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model.config.use_cache = True
    return model.to(args.device).eval(), tokenizer


def main(argv=None):
    args = parse_args(argv)
    distributed = args.num_checkpoint_shards > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed LongBench evaluation requires CUDA")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    bench.validate_args(args)
    tasks = bench.selected_tasks(args.tasks)
    budgets = [args.v2_max_new_tokens] if args.benchmark in ("longbench-v2", "both") else []
    if args.benchmark in ("longbench", "both"):
        budgets.extend(bench.LONGBENCH_MAX_NEW_TOKENS[t] for t in tasks)
    if args.max_context_tokens <= max(budgets):
        raise ValueError("Context budget must exceed generation budget")
    config, is_v4 = model_plan(args)
    dtype = bench.resolve_dtype(args.dtype, args.device)
    manifest = {
        "entrypoint": "lyra.eval_longbench", "checkpoint": args.checkpoint.name,
        "base_model": args.base_model.name if args.base_model else getattr(config, "tvmf_base_model", None),
        "benchmark": args.benchmark, "tasks": tasks, "max_samples": args.max_samples,
        "max_context_tokens": args.max_context_tokens, "v2_max_new_tokens": args.v2_max_new_tokens,
        "longbench": args.longbench.name, "longbench_v2": args.longbench_v2.name,
        "dtype": str(dtype), "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "tvmf": attention_recipe(config) if is_v4 else None,
        "rope_parameters": getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None),
        "enable_thinking": False, "middle_truncation": True,
        "generation_constraints": {"passage_retrieval_en": "Paragraph 1..30", "longbench_v2": "The correct answer is ([A-D])"},
        "contaminated_training_benchmark": getattr(config, "tvmf_benchmark_contamination", None) if is_v4 else False,
        "warning": "Check training-data provenance before interpreting these scores as held-out generalization.",
    }
    if args.checkpoint_shard == 0:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        if args.benchmark in ("longbench", "both"):
            next(bench.iter_longbench(args.longbench, tasks[0], 1))
        if args.benchmark in ("longbench-v2", "both"):
            next(bench.iter_longbench_v2(args.longbench_v2, 1))
        if args.checkpoint_shard == 0:
            print("Dry run passed: config/data readable; no weights loaded or results written.")
        if distributed:
            torch.distributed.destroy_process_group()
        return
    run_dir = args.output_dir / bench.checkpoint_label(args.checkpoint)
    if args.checkpoint_shard == 0:
        bench.prepare_run_dir(run_dir, manifest, args.overwrite)
    if distributed:
        torch.distributed.barrier()
    try:
        model, tokenizer = load_model(args, config, is_v4, dtype)
        if args.benchmark in ("longbench", "both"):
            bench.evaluate_v1(model, tokenizer, run_dir, tasks, args)
        if args.benchmark in ("longbench-v2", "both"):
            bench.evaluate_v2(model, tokenizer, run_dir, args)
        if distributed:
            torch.distributed.barrier()
            if args.checkpoint_shard == 0:
                bench.merge_distributed_results(run_dir, tasks, args)
            torch.distributed.barrier()
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
