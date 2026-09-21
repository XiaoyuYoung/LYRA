"""Per-query-head learnable t-vMF attention for Hugging Face Qwen3.

The similarity transform is

    (1 + cos) / (1 + kappa * (1 - cos)) - 1

and is applied to post-RoPE query/key vectors. Query chunking bounds peak
score-matrix memory; this is especially important for the 32 heads in Qwen3-8B.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import nn


DEFAULT_KAPPA = 4.0
BACKEND_NAME = "tvmf_v4"
ScaleMode = Literal["qk_norm", "sqrt_head_dim", "none"]
SimilarityMode = Literal["raw", "calibrated"]
LayerSpec = str | int | Iterable[int]


class TvmfHeadParameters(nn.Module):
    """Positive scalars for each query head, kept in FP32 even for BF16 models."""

    def __init__(self, heads: int, kappa: float = DEFAULT_KAPPA, head_scale: float = 1.0,
                 learnable_kappa: bool = True, learnable_scale: bool = True, device=None):
        super().__init__()
        if not math.isfinite(kappa) or kappa < 0 or (learnable_kappa and kappa == 0):
            raise ValueError("kappa must be finite and positive when learnable (fixed kappa may be zero)")
        if not math.isfinite(head_scale) or head_scale <= 0:
            raise ValueError("head_scale must be finite and positive")
        self.initial_kappa, self.initial_scale = kappa, head_scale
        self.learnable_kappa, self.learnable_scale = learnable_kappa, learnable_scale
        for name, initial, learnable in (("kappa", kappa, learnable_kappa),
                                         ("scale", head_scale, learnable_scale)):
            value = self.inverse_softplus(initial) if learnable else initial
            tensor = torch.full((heads,), value, dtype=torch.float32, device=device)
            if learnable:
                self.register_parameter("raw_" + name, nn.Parameter(tensor))
            else:
                self.register_buffer("fixed_" + name, tensor)

    @staticmethod
    def inverse_softplus(value: float) -> float:
        return value + math.log(-math.expm1(-value))

    def reset_parameters(self):
        for name, initial, learnable in (("kappa", self.initial_kappa, self.learnable_kappa),
                                         ("scale", self.initial_scale, self.learnable_scale)):
            tensor = getattr(self, ("raw_" if learnable else "fixed_") + name)
            nn.init.constant_(tensor, self.inverse_softplus(initial) if learnable else initial)

    def _apply(self, fn, recurse=True):
        def keep_fp32(tensor):
            converted = fn(tensor)
            if converted.is_floating_point() and converted.dtype != torch.float32:
                # Cast directly from the original so model.bfloat16() cannot round
                # the FP32 master parameters before converting them back to float.
                if tensor.is_meta:
                    return converted.float()
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(keep_fp32, recurse=recurse)

    def _positive(self, name, learnable):
        if not learnable:
            return getattr(self, "fixed_" + name)
        return nn.functional.softplus(getattr(self, "raw_" + name).float()) + torch.finfo(torch.float32).tiny

    def forward(self):
        return (self._positive("kappa", self.learnable_kappa).view(1, -1, 1, 1),
                self._positive("scale", self.learnable_scale).view(1, -1, 1, 1))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def tvmf_similarity(cosine: torch.Tensor, kappa: torch.Tensor | float,
                    mode: SimilarityMode = "calibrated") -> torch.Tensor:
    """Raw f(c), or (f(c)-f(0))/f'(0); tensor kappa stays on the autograd graph."""
    denominator = 1.0 + (1.0 - cosine) * kappa
    if mode == "raw":
        return (1.0 + cosine) / denominator - 1.0
    if mode == "calibrated":
        return ((1.0 + kappa) / denominator) * cosine
    raise ValueError(f"Unsupported similarity mode: {mode}")


def _fixed_logit_scale(scaling: float, scale_mode: ScaleMode, logit_scale: float | None) -> float:
    if logit_scale is not None:
        return float(logit_scale)
    if scale_mode == "none":
        return 1.0
    if scale_mode == "qk_norm":
        raise ValueError("qk_norm scale is pair-dependent and is handled in the attention function.")
    if scale_mode == "sqrt_head_dim":
        return 1.0 / float(scaling)
    raise ValueError(f"Unsupported scale_mode: {scale_mode}")


def resolve_layer_indices(layers: LayerSpec, num_hidden_layers: int) -> tuple[int, ...]:
    """Resolve ``all``, ``last``, comma-separated, integer, or negative indices."""
    if isinstance(layers, str):
        value = layers.strip().lower()
        if value == "all":
            return tuple(range(num_hidden_layers))
        if value == "last":
            return (num_hidden_layers - 1,)
        if not value:
            return ()
        raw_indices = [int(part.strip()) for part in value.split(",")]
    elif isinstance(layers, int):
        raw_indices = [layers]
    else:
        raw_indices = [int(index) for index in layers]

    resolved = []
    for index in raw_indices:
        index = index + num_hidden_layers if index < 0 else index
        if not 0 <= index < num_hidden_layers:
            raise ValueError(f"Layer index {index} is outside [0, {num_hidden_layers - 1}].")
        resolved.append(index)
    return tuple(sorted(set(resolved)))


def _uses_tvmf(module: nn.Module) -> bool:
    configured = getattr(module.config, "tvmf_layer_indices", None)
    return configured is None or module.layer_idx in configured


def _native_eager_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    return torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous(), attn_weights


def _native_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
    kwargs: dict[str, Any],
):
    backend = getattr(module.config, "tvmf_native_backend", "eager")
    if backend == "sdpa":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )
    if backend != "eager":
        raise ValueError(f"Unsupported native fallback backend: {backend}")
    return _native_eager_attention(module, query, key, value, attention_mask, scaling, dropout)


def _causal_mask_for_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    query_start: int,
    full_query_len: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Create a causal mask when the SDPA-style mask was elided."""
    query_len = query.shape[-2]
    key_len = key.shape[-2]
    absolute_query = (
        torch.arange(query_start, query_start + query_len, device=query.device)
        + key_len
        - full_query_len
    )
    key_position = torch.arange(key_len, device=query.device)
    allowed = key_position[None, :] <= absolute_query[:, None]
    if sliding_window is not None:
        allowed &= key_position[None, :] > (absolute_query[:, None] - int(sliding_window))
    return allowed[None, None, :, :]


def tvmf_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    scale_mode: ScaleMode | None = None,
    logit_scale: float | None = None,
    eps: float = 1e-6,
    **kwargs: Any,
):
    """Qwen3-compatible t-vMF attention."""
    scale_mode = scale_mode or getattr(module.config, "tvmf_scale_mode", "qk_norm")
    logit_scale = logit_scale if logit_scale is not None else getattr(module.config, "tvmf_logit_scale", None)

    uses_tvmf = _uses_tvmf(module)
    if not uses_tvmf:
        return _native_attention(module, query, key, value, attention_mask, scaling, dropout, dict(kwargs))

    if not hasattr(module, "tvmf_parameters"):
        raise RuntimeError("Missing v4 head parameters; construct with V4Qwen3ForCausalLM")
    kappa, head_scale = module.tvmf_parameters()
    if kappa.shape[1] != query.shape[1]:
        raise ValueError("v4 requires one kappa and scale per query head, not per KV head")
    similarity_mode = module.config.tvmf_similarity_mode

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    query_float = query.float()
    key_float = key_states.float()
    query_norm = torch.linalg.vector_norm(query_float, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    key_norm = torch.linalg.vector_norm(key_float, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    query_unit = query_float / query_norm
    key_unit = key_float / key_norm

    query_len = query.shape[-2]
    chunk_size = int(getattr(module.config, "tvmf_query_chunk_size", 0) or query_len)
    chunk_size = min(max(chunk_size, 1), query_len)
    output_attentions = bool(kwargs.get("output_attentions", False))
    checkpoint_chunks = bool(getattr(module.config, "tvmf_checkpoint_chunks", False))
    sliding_window = kwargs.get("sliding_window")

    def attend_chunk(
        q_unit: torch.Tensor,
        q_norm: torch.Tensor,
        k_unit: torch.Tensor,
        k_norm: torch.Tensor,
        values: torch.Tensor,
        chunk_kappa: torch.Tensor,
        chunk_scale: torch.Tensor,
        mask: torch.Tensor | None,
        query_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = torch.matmul(q_unit, k_unit.transpose(2, 3)).clamp(-1.0, 1.0)
        weights = tvmf_similarity(cosine, chunk_kappa, similarity_mode)
        if scale_mode == "qk_norm" and logit_scale is None:
            weights = weights * (q_norm * k_norm.transpose(2, 3) * float(scaling))
        else:
            weights = weights * _fixed_logit_scale(scaling, scale_mode, logit_scale)
        weights = weights * chunk_scale

        if mask is None and getattr(module, "is_causal", True):
            allowed = _causal_mask_for_chunk(q_unit, k_unit, query_start, query_len, sliding_window)
            weights = weights.masked_fill(~allowed, torch.finfo(weights.dtype).min)
        elif mask is not None:
            if mask.dtype == torch.bool:
                weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
            else:
                weights = weights + mask
        weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = nn.functional.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, values).transpose(1, 2).contiguous()
        return output, weights

    outputs: list[torch.Tensor] = []
    attention_chunks: list[torch.Tensor] = []
    for start in range(0, query_len, chunk_size):
        end = min(start + chunk_size, query_len)
        mask_chunk = attention_mask
        if attention_mask is not None and attention_mask.shape[-2] != 1:
            mask_chunk = attention_mask[..., start:end, :]
        tensor_args = (
            query_unit[..., start:end, :],
            query_norm[..., start:end, :],
            key_unit,
            key_norm,
            value_states,
            kappa,
            head_scale,
        )
        should_checkpoint = (
            checkpoint_chunks
            and module.training
            and torch.is_grad_enabled()
            and not output_attentions
            and any(tensor.requires_grad for tensor in tensor_args)
        )
        if should_checkpoint:
            from torch.utils.checkpoint import checkpoint

            def output_only(
                *args: torch.Tensor,
                _mask: torch.Tensor | None = mask_chunk,
                _start: int = start,
            ) -> torch.Tensor:
                return attend_chunk(*args, _mask, _start)[0]

            chunk_output = checkpoint(output_only, *tensor_args, use_reentrant=False)
            chunk_weights = None
        else:
            chunk_output, chunk_weights = attend_chunk(*tensor_args, mask_chunk, start)
        outputs.append(chunk_output)
        if output_attentions and chunk_weights is not None:
            attention_chunks.append(chunk_weights)

    attn_output = torch.cat(outputs, dim=1)
    attn_weights = torch.cat(attention_chunks, dim=-2) if attention_chunks else None
    return attn_output, attn_weights


def register_tvmf_attention() -> None:
    """Use a separate registry name so importing v4 never replaces v3's backend."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

    ALL_ATTENTION_FUNCTIONS.register(BACKEND_NAME, tvmf_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(BACKEND_NAME, sdpa_mask)
