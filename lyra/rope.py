"""Shared static YaRN configuration for training, loading and evaluation."""

import copy
import math

ORIGINAL_CONTEXT = 32768


def yarn_config(config, factor):
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError("YaRN factor must be finite and greater than 1")
    result = copy.deepcopy(config)
    current = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if "full_attention" in current:
        raise ValueError("Per-layer RoPE configs are unsupported by this Qwen3-8B entrypoint")
    theta = current.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
    rope = {"rope_type": "yarn", "factor": float(factor), "original_max_position_embeddings": ORIGINAL_CONTEXT}
    result.max_position_embeddings = int(ORIGINAL_CONTEXT * factor)
    if hasattr(config, "rope_parameters"):
        result.rope_parameters = {**rope, "rope_theta": float(theta)}
    else:
        result.rope_scaling, result.rope_theta = rope, float(theta)
    return result


def training_rope_config(config, max_length, yarn_factor=None, *, resume=False):
    current = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    is_yarn = current.get("rope_type", current.get("type")) == "yarn"
    if resume:
        if yarn_factor is not None and (not is_yarn or current.get("factor") != yarn_factor):
            raise ValueError("Cannot change YaRN factor while resuming optimizer state; use the saved factor")
        result = copy.deepcopy(config)
    else:
        result = yarn_config(config, yarn_factor) if yarn_factor is not None else copy.deepcopy(config)
    rope = getattr(result, "rope_parameters", None) or getattr(result, "rope_scaling", None) or {}
    capacity = result.max_position_embeddings if rope.get("rope_type", rope.get("type")) == "yarn" else ORIGINAL_CONTEXT
    if max_length > capacity:
        raise ValueError(f"Training length {max_length} exceeds RoPE capacity {capacity}; use --yarn-factor 2 for 64K or 4 for 128K")
    return result
