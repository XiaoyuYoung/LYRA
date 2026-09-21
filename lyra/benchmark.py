"""LongBench prompts, constraints, and prediction-record utilities."""

from __future__ import annotations

import argparse
import gc
import io
import json
import re
import sys
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



DEFAULT_LONGBENCH = PROJECT_ROOT / "external_data" / "LongBench" / "data.zip"
DEFAULT_LONGBENCH_V2 = PROJECT_ROOT / "external_data" / "LongBench-v2" / "data.json"

LONGBENCH_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)

# The official LongBench runner intentionally does not wrap these few-shot and
# code-completion prompts in a model chat template.
RAW_PROMPT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

LONGBENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 64,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

if __package__:
    from .prompts import LONGBENCH_PROMPTS, LONGBENCH_V2_PROMPT
else:
    from prompts import LONGBENCH_PROMPTS, LONGBENCH_V2_PROMPT


PASSAGE_RETRIEVAL_EN_RESPONSES = tuple(f"Paragraph {number}" for number in range(1, 31))
LONGBENCH_V2_RESPONSES = tuple(f"The correct answer is ({letter})" for letter in "ABCD")


def checkpoint_label(checkpoint: Path) -> str:
    return checkpoint.name


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without materializing the 465 MB v2 file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = buffer == ""
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer) and not eof:
                    continue
                if position >= len(buffer) or buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                position += 1
                started = True
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    position = end
                    if not isinstance(item, dict):
                        raise ValueError(f"Expected objects in {path}, got {type(item).__name__}")
                    yield item
                    if position > chunk_size:
                        buffer = buffer[position:]
                        position = 0
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer = buffer[position:] + handle.read(chunk_size)
                    position = 0
                    eof = handle.tell() == path.stat().st_size


def iter_longbench(zip_path: Path, task: str, limit: int | None) -> Iterator[dict[str, Any]]:
    member = f"data/{task}.jsonl"
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            raise FileNotFoundError(f"{member} is missing from {zip_path}")
        with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if limit is not None and index >= limit:
                    break
                item = json.loads(line)
                item["_eval_id"] = f"{task}:{index}"
                item["_eval_index"] = index
                yield item


def iter_longbench_v2(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    for index, item in enumerate(iter_json_array(path)):
        if limit is not None and index >= limit:
            break
        item["_eval_index"] = index
        yield item


def render_chat(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        token_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def prepare_inputs(
    tokenizer,
    prompt: str,
    max_context_tokens: int,
    max_new_tokens: int,
    device: str,
    *,
    use_chat_template: bool,
):
    budget = max_context_tokens - max_new_tokens
    if budget <= 0:
        raise ValueError("max-context-tokens must be larger than max-new-tokens")
    token_ids = render_chat(tokenizer, prompt) if use_chat_template else tokenizer.encode(prompt)
    original_tokens = len(token_ids)
    truncated = original_tokens > budget
    if truncated:
        left = budget // 2
        token_ids = token_ids[:left] + token_ids[-(budget - left) :]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, original_tokens, truncated


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def extract_v2_answer(response: str) -> str | None:
    cleaned = response.replace("*", "").strip()
    patterns = (
        r"the correct answer is\s*\(?([A-D])\)?",
        r"(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?(?:[.。,:：\s]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def build_candidate_constraint(
    tokenizer,
    candidates: Sequence[str],
    *,
    prompt_length: int,
    eos_token_id: int,
) -> tuple[Callable[[int, torch.Tensor], list[int]], dict[tuple[int, ...], str]]:
    """Build a token-prefix constraint that can finish only as one candidate.

    Candidate text is tokenized exactly as it will appear after the generation
    prompt. Once a complete candidate has been emitted, only EOS is legal.
    """
    encoded: dict[tuple[int, ...], str] = {}
    for candidate in candidates:
        token_ids = tuple(tokenizer.encode(candidate, add_special_tokens=False))
        if not token_ids:
            raise ValueError(f"Candidate tokenized to an empty sequence: {candidate!r}")
        if eos_token_id in token_ids:
            raise ValueError(f"Candidate unexpectedly contains EOS: {candidate!r}")
        previous = encoded.setdefault(token_ids, candidate)
        if previous != candidate:
            raise ValueError(f"Candidates tokenize identically: {previous!r} and {candidate!r}")

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = tuple(int(token_id) for token_id in input_ids[prompt_length:].tolist())
        matching = [token_ids for token_ids in encoded if token_ids[: len(generated)] == generated]
        if not matching:
            raise RuntimeError(f"Generated prefix escaped outside the candidate set: {generated}")
        allowed = {
            token_ids[len(generated)] if len(generated) < len(token_ids) else eos_token_id
            for token_ids in matching
        }
        return sorted(allowed)

    return allowed_tokens, encoded


def generation_hit_token_limit(
    generated_ids: Sequence[int],
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> bool:
    """Return true only when generation exhausted its budget without EOS."""
    return len(generated_ids) >= max_new_tokens and (
        not generated_ids or int(generated_ids[-1]) not in eos_token_ids
    )


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
                ids.add(str(item["id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid result at {path}:{line_number}: {exc}") from exc
    return ids


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    args: argparse.Namespace,
    *,
    use_chat_template: bool = True,
    extra_eos_token_id: int | None = None,
    candidate_responses: Sequence[str] | None = None,
):
    primary_eos_token_id = tokenizer.eos_token_id
    if candidate_responses and primary_eos_token_id is None:
        raise ValueError("Constrained generation requires tokenizer.eos_token_id")

    # Constrained responses cannot consume the full nominal generation limit.
    # Reserve their actual maximum length so the unused budget remains available
    # to the input context.
    generation_reserve = max_new_tokens
    if candidate_responses:
        candidate_lengths = [
            len(tokenizer.encode(candidate, add_special_tokens=False)) for candidate in candidate_responses
        ]
        generation_reserve = max(candidate_lengths) + 1  # Include EOS.
        if generation_reserve > max_new_tokens:
            raise ValueError(
                f"Longest constrained response needs {generation_reserve} tokens including EOS, "
                f"above max_new_tokens={max_new_tokens}"
            )
    inputs, original_tokens, truncated = prepare_inputs(
        tokenizer,
        prompt,
        args.max_context_tokens,
        generation_reserve,
        args.device,
        use_chat_template=use_chat_template,
    )
    input_tokens = int(inputs["input_ids"].shape[1])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.perf_counter()
    eos_token_id: int | list[int] | None = primary_eos_token_id
    if extra_eos_token_id is not None:
        eos_values = [value for value in (primary_eos_token_id, extra_eos_token_id) if value is not None]
        eos_token_id = list(dict.fromkeys(eos_values))
    generation_kwargs: dict[str, Any] = {}
    candidate_token_map: dict[tuple[int, ...], str] | None = None
    if candidate_responses:
        constraint, candidate_token_map = build_candidate_constraint(
            tokenizer,
            candidate_responses,
            prompt_length=input_tokens,
            eos_token_id=primary_eos_token_id,
        )
        generation_kwargs["prefix_allowed_tokens_fn"] = constraint
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **generation_kwargs,
        )
    elapsed = time.perf_counter() - started
    new_ids = output[0, input_tokens:]
    generated_ids = [int(token_id) for token_id in new_ids.tolist()]
    eos_token_ids = {
        int(value)
        for value in ([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
    }
    output_truncated = generation_hit_token_limit(generated_ids, max_new_tokens, eos_token_ids)
    response_ids = tuple(token_id for token_id in generated_ids if token_id not in eos_token_ids)
    if candidate_token_map is not None:
        response = candidate_token_map.get(response_ids)
        if response is None:
            raise RuntimeError(f"Constrained generation did not finish as a candidate: {response_ids}")
    else:
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response, input_tokens, original_tokens, truncated, output_truncated, int(new_ids.numel()), elapsed


def write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def distributed_output_path(path: Path, args: argparse.Namespace) -> Path:
    if args.num_checkpoint_shards == 1:
        return path
    return path.with_name(
        f"{path.stem}.part-{args.checkpoint_shard:05d}-of-{args.num_checkpoint_shards:05d}{path.suffix}"
    )


def distributed_completed_ids(path: Path, part_path: Path) -> set[str]:
    done = completed_ids(path)
    if part_path != path:
        done.update(completed_ids(part_path))
    return done


def evaluate_v1(model, tokenizer, run_dir: Path, tasks: list[str], args: argparse.Namespace) -> None:
    output_dir = run_dir / "longbench"
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        merged_path = output_dir / f"{task}.jsonl"
        output_path = distributed_output_path(merged_path, args)
        done = distributed_completed_ids(merged_path, output_path)
        seen = 0
        with output_path.open("a", encoding="utf-8") as handle:
            for item in iter_longbench(args.longbench, task, args.max_samples):
                if item["_eval_index"] % args.num_checkpoint_shards != args.checkpoint_shard:
                    continue
                record_id = item["_eval_id"]
                if record_id in done:
                    continue
                prompt = LONGBENCH_PROMPTS[task].format(context=item["context"], input=item["input"])
                max_new_tokens = LONGBENCH_MAX_NEW_TOKENS[task]
                newline_eos = None
                if task == "samsum":
                    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                    newline_eos = newline_ids[-1] if newline_ids else None
                candidate_responses = (
                    PASSAGE_RETRIEVAL_EN_RESPONSES if task == "passage_retrieval_en" else None
                )
                response, n_input, n_original, truncated, output_truncated, n_new, elapsed = generate_one(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    args,
                    use_chat_template=task not in RAW_PROMPT_TASKS,
                    extra_eos_token_id=newline_eos,
                    candidate_responses=candidate_responses,
                )
                write_record(
                    handle,
                    {
                        "id": record_id,
                        "index": item["_eval_index"],
                        "benchmark": "longbench",
                        "dataset": task,
                        "pred": response,
                        "answers": item["answers"],
                        "all_classes": item.get("all_classes"),
                        "length": item.get("length"),
                        "language": item.get("language"),
                        "prompt_mode": "raw" if task in RAW_PROMPT_TASKS else "chat_template",
                        "input_tokens": n_input,
                        "original_input_tokens": n_original,
                        "input_truncated": truncated,
                        "truncated": truncated,
                        "output_truncated": output_truncated,
                        "output_constraint": "Paragraph 1..30" if candidate_responses else None,
                        "new_tokens": n_new,
                        "elapsed_seconds": round(elapsed, 4),
                    },
                )
                seen += 1
                if args.log_every > 0 and seen % args.log_every == 0:
                    print(f"[{run_dir.name}] {task}: generated {seen} new predictions", flush=True)
        print(f"[{run_dir.name}] {task}: complete ({seen} new, {len(done)} resumed)", flush=True)


def evaluate_v2(model, tokenizer, run_dir: Path, args: argparse.Namespace) -> None:
    merged_path = run_dir / "longbench_v2.jsonl"
    output_path = distributed_output_path(merged_path, args)
    done = distributed_completed_ids(merged_path, output_path)
    seen = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for item in iter_longbench_v2(args.longbench_v2, args.max_samples):
            if item["_eval_index"] % args.num_checkpoint_shards != args.checkpoint_shard:
                continue
            record_id = str(item["_id"])
            if record_id in done:
                continue
            prompt = LONGBENCH_V2_PROMPT.format(**item)
            response, n_input, n_original, truncated, output_truncated, n_new, elapsed = generate_one(
                model,
                tokenizer,
                prompt,
                args.v2_max_new_tokens,
                args,
                candidate_responses=LONGBENCH_V2_RESPONSES,
            )
            pred = extract_v2_answer(response)
            write_record(
                handle,
                {
                    "id": record_id,
                    "index": item["_eval_index"],
                    "benchmark": "longbench-v2",
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                    "answer": item["answer"],
                    "response": response,
                    "pred": pred,
                    "judge": pred == item["answer"],
                    "context_chars": len(item["context"]),
                    "input_tokens": n_input,
                    "original_input_tokens": n_original,
                    "input_truncated": truncated,
                    "truncated": truncated,
                    "output_truncated": output_truncated,
                    "output_constraint": "The correct answer is ([A-D])",
                    "new_tokens": n_new,
                    "elapsed_seconds": round(elapsed, 4),
                },
            )
            seen += 1
            if args.log_every > 0 and seen % args.log_every == 0:
                print(f"[{run_dir.name}] LongBench-v2: generated {seen} new predictions", flush=True)
    print(f"[{run_dir.name}] LongBench-v2: complete ({seen} new, {len(done)} resumed)", flush=True)


def merge_result_parts(path: Path, num_shards: int) -> None:
    part_paths = [
        path.with_name(f"{path.stem}.part-{rank:05d}-of-{num_shards:05d}{path.suffix}")
        for rank in range(num_shards)
    ]
    records: dict[str, dict[str, Any]] = {}
    for source in [path, *part_paths]:
        if not source.is_file():
            continue
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    record_id = str(record["id"])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"Invalid result at {source}:{line_number}: {exc}") from exc
                previous = records.setdefault(record_id, record)
                if previous != record:
                    raise ValueError(f"Conflicting distributed result for id {record_id!r}")
    temporary = path.with_name(f".{path.name}.merge.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in sorted(records.values(), key=lambda item: (int(item.get("index", 0)), str(item["id"]))):
            write_record(handle, record)
    temporary.replace(path)
    for part_path in part_paths:
        if part_path.exists():
            part_path.unlink()


def merge_distributed_results(run_dir: Path, tasks: list[str], args: argparse.Namespace) -> None:
    if args.num_checkpoint_shards == 1:
        return
    if args.benchmark in ("longbench", "both"):
        for task in tasks:
            merge_result_parts(run_dir / "longbench" / f"{task}.jsonl", args.num_checkpoint_shards)
    if args.benchmark in ("longbench-v2", "both"):
        merge_result_parts(run_dir / "longbench_v2.jsonl", args.num_checkpoint_shards)


def selected_tasks(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(LONGBENCH_TASKS)
    unknown = sorted(set(values) - set(LONGBENCH_TASKS))
    if unknown:
        raise ValueError(f"Unknown LongBench task(s): {unknown}")
    return list(dict.fromkeys(values))


def validate_args(args: argparse.Namespace) -> None:
    if args.max_context_tokens <= 0 or args.v2_max_new_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.num_checkpoint_shards <= 0 or not 0 <= args.checkpoint_shard < args.num_checkpoint_shards:
        raise ValueError("Require 0 <= checkpoint-shard < num-checkpoint-shards")
    if args.benchmark in ("longbench", "both") and not args.longbench.is_file():
        raise FileNotFoundError(args.longbench)
    if args.benchmark in ("longbench-v2", "both") and not args.longbench_v2.is_file():
        raise FileNotFoundError(args.longbench_v2)
    if args.device.startswith("cuda") and not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on a GPU node or pass --device cpu")


def prepare_run_dir(run_dir: Path, config: dict[str, Any], overwrite: bool) -> None:
    manifest = run_dir / "run_config.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        runtime_keys = {"started_at", "checkpoint_kind"}
        comparable = {key: value for key, value in previous.items() if key not in runtime_keys}
        current = {key: value for key, value in config.items() if key not in runtime_keys}
        if comparable != current and not overwrite:
            raise RuntimeError(
                f"{run_dir} contains results produced with different settings. "
                "Choose another --output-dir or pass --overwrite."
            )
    if overwrite and run_dir.is_dir():
        for path in run_dir.rglob("*.jsonl"):
            path.unlink()
        for name in ("metrics.json",):
            path = run_dir / name
            if path.exists():
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
