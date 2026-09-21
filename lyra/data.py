"""Local long-context SFT dataset adapters and response-only collation."""

from __future__ import annotations

import bisect
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize common LongAlign/LongCite/ShareGPT/QA schemas."""
    raw_messages = row.get("messages") or row.get("conversations")
    if raw_messages:
        messages = []
        for message in raw_messages:
            raw_role = str(message.get("role", message.get("from", ""))).lower()
            role = ROLE_MAP.get(raw_role)
            content = message.get("content", message.get("value"))
            if role is None or content is None:
                raise ValueError(f"Unsupported conversation message: {message}")
            messages.append({"role": role, "content": str(content)})
        return messages

    answer = row.get("answer", row.get("output", row.get("response")))
    if answer is not None:
        if row.get("context") is not None:
            question = row.get("question", row.get("instruction", "请根据上下文回答问题。"))
            prompt = f"上下文：\n{row['context']}\n\n问题：\n{question}"
        else:
            instruction = str(row.get("instruction", row.get("question", "")))
            extra_input = str(row.get("input", ""))
            prompt = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(answer)}]

    if row.get("text") is not None:
        return [{"role": "user", "content": str(row["text"])}]
    raise ValueError(f"Cannot infer a supported text schema from columns: {sorted(row)}")


class LongContextSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        max_length: int,
        max_samples: int | None = None,
        response_only: bool = True,
        cache_dir: str | None = None,
        overflow_strategy: str = "error",
    ) -> None:
        filename = Path(path).name.lower()
        if not filename.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
            raise ValueError("Training data must be a local JSON/JSONL file, optionally gzip-compressed.")
        self.rows = load_dataset("json", data_files=path, split="train", cache_dir=cache_dir)
        if max_samples is not None:
            self.rows = self.rows.select(range(min(max_samples, len(self.rows))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_only = response_only
        self.overflow_strategy = overflow_strategy

    def __len__(self) -> int:
        return len(self.rows)

    def _chat_ids(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
        kwargs = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
        try:
            encoded = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = row_to_messages(dict(self.rows[index]))
        has_target = len(messages) >= 2 and messages[-1]["role"] == "assistant"
        if has_target:
            full_ids = self._chat_ids(messages, add_generation_prompt=False)
            prefix_ids = self._chat_ids(messages[:-1], add_generation_prompt=True)
            prefix_length = 0
            for left, right in zip(prefix_ids, full_ids):
                if left != right:
                    break
                prefix_length += 1
        else:
            full_ids = self.tokenizer(messages[0]["content"], add_special_tokens=True)["input_ids"]
            prefix_length = 0

        overflow = max(0, len(full_ids) - self.max_length)
        if overflow:
            if self.overflow_strategy == "error":
                raise ValueError(
                    f"Sample {index} has {len(full_ids)} tokens, above max_length={self.max_length}. "
                    "Use shards prepared for this limit or explicitly select truncate_left."
                )
            if self.overflow_strategy != "truncate_left":
                raise ValueError(f"Unknown overflow strategy: {self.overflow_strategy}")
            full_ids = full_ids[overflow:]
            prefix_length = max(0, prefix_length - overflow)

        labels = list(full_ids)
        if self.response_only and has_target:
            labels[:prefix_length] = [-100] * prefix_length
        if not any(label != -100 for label in labels):
            raise ValueError(f"Sample {index} has no supervised tokens after tokenization/truncation.")
        return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


class MultiSourceSFTDataset(torch.utils.data.Dataset):
    """Concatenate shards or sample them with explicit source probabilities."""

    def __init__(
        self,
        paths: list[str],
        tokenizer,
        *,
        max_length: int,
        response_only: bool = True,
        cache_dir: str | None = None,
        source_weights: list[float] | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 42,
        max_samples: int | None = None,
        overflow_strategy: str = "error",
    ) -> None:
        if not paths:
            raise ValueError("At least one data shard is required.")
        self.datasets = [
            LongContextSFTDataset(
                path,
                tokenizer,
                max_length=max_length,
                response_only=response_only,
                cache_dir=cache_dir,
                overflow_strategy=overflow_strategy,
            )
            for path in paths
        ]
        self.lengths = [len(dataset) for dataset in self.datasets]
        if any(length == 0 for length in self.lengths):
            raise ValueError(f"Empty data shard detected: {dict(zip(paths, self.lengths))}")
        self.cumulative_lengths = []
        running = 0
        for length in self.lengths:
            running += length
            self.cumulative_lengths.append(running)

        self.seed = seed
        self.source_weights = None
        if source_weights is not None:
            if len(source_weights) != len(paths):
                raise ValueError("--source-weights must contain one value per --train-file shard.")
            if any(weight < 0 for weight in source_weights) or sum(source_weights) <= 0:
                raise ValueError("Source weights must be non-negative with a positive sum.")
            total = sum(source_weights)
            normalized = [weight / total for weight in source_weights]
            cumulative, running_weight = [], 0.0
            for weight in normalized:
                running_weight += weight
                cumulative.append(running_weight)
            cumulative[-1] = 1.0
            self.source_weights = cumulative

        self.epoch_length = samples_per_epoch or sum(self.lengths)
        if max_samples is not None:
            self.epoch_length = min(self.epoch_length, max_samples)

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if self.source_weights is None:
            source_index = bisect.bisect_right(self.cumulative_lengths, index)
            previous = 0 if source_index == 0 else self.cumulative_lengths[source_index - 1]
            return self.datasets[source_index][index - previous]

        rng = random.Random(self.seed + index)
        source_index = min(bisect.bisect_right(self.source_weights, rng.random()), len(self.datasets) - 1)
        return self.datasets[source_index][rng.randrange(self.lengths[source_index])]


@dataclass
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_length - length
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * length + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
