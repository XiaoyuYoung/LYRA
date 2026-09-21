"""Qwen3 t-vMF construction and lossless full/compact checkpoint loading."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3PreTrainedModel

if __package__:
    from .tvmf_attention import BACKEND_NAME, DEFAULT_KAPPA, TvmfHeadParameters, register_tvmf_attention, resolve_layer_indices
else:
    from tvmf_attention import BACKEND_NAME, DEFAULT_KAPPA, TvmfHeadParameters, register_tvmf_attention, resolve_layer_indices


CONFIG_FILENAME = "tvmf_v4_config.json"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT = str(Path(__file__).resolve().parents[1] / "outputs" / "lyra-qwen3-8b")
QWEN3_8B_SIGNATURE = {
    "model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
    "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8,
    "head_dim": 128, "vocab_size": 151936,
}


def model_signature(config):
    return {name: getattr(config, name, None) for name in QWEN3_8B_SIGNATURE}


def validate_qwen3_8b(config):
    actual = model_signature(config)
    mismatches = {k: {"expected": v, "actual": actual[k]} for k, v in QWEN3_8B_SIGNATURE.items() if actual[k] != v}
    if mismatches:
        raise ValueError(f"Expected Qwen3-8B architecture: {mismatches}")


def register_backend():
    register_tvmf_attention()


def v4_config(config, *, kappa=DEFAULT_KAPPA, head_scale=1.0, learnable_kappa=True,
              learnable_scale=True, similarity_mode="calibrated", scale_mode="qk_norm",
              logit_scale=None, layers="last", query_chunk_size=256, checkpoint_chunks=True):
    """Prepare parameters BEFORE from_pretrained builds its expected state dict."""
    if not math.isfinite(kappa) or kappa < 0 or (learnable_kappa and kappa == 0):
        raise ValueError("kappa must be positive when learnable, or non-negative when fixed")
    if not math.isfinite(head_scale) or head_scale <= 0:
        raise ValueError("head_scale must be finite and positive")
    if similarity_mode not in ("raw", "calibrated") or scale_mode not in ("qk_norm", "sqrt_head_dim", "none"):
        raise ValueError("Invalid similarity_mode or scale_mode")
    if logit_scale is not None and (not math.isfinite(logit_scale) or logit_scale <= 0):
        raise ValueError("logit_scale must be finite and positive")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    result = copy.deepcopy(config)
    result.tvmf_version = 4
    result.tvmf_kappa = float(kappa)
    result.tvmf_head_scale = float(head_scale)
    result.tvmf_learnable_kappa = bool(learnable_kappa)
    result.tvmf_learnable_scale = bool(learnable_scale)
    result.tvmf_similarity_mode = similarity_mode
    result.tvmf_scale_mode = scale_mode
    result.tvmf_logit_scale = logit_scale
    result.tvmf_layer_indices = list(resolve_layer_indices(layers, config.num_hidden_layers))
    if not result.tvmf_layer_indices:
        raise ValueError("Select at least one t-vMF layer")
    result.tvmf_query_chunk_size = int(query_chunk_size)
    result.tvmf_checkpoint_chunks = bool(checkpoint_chunks)
    result.tvmf_native_backend = "sdpa"
    result._attn_implementation = BACKEND_NAME
    return result


def attention_recipe(config):
    return {
        "kappa": config.tvmf_kappa, "head_scale": config.tvmf_head_scale,
        "learnable_kappa": config.tvmf_learnable_kappa, "learnable_scale": config.tvmf_learnable_scale,
        "similarity_mode": config.tvmf_similarity_mode, "scale_mode": config.tvmf_scale_mode,
        "logit_scale": config.tvmf_logit_scale, "layers": config.tvmf_layer_indices,
        "query_chunk_size": config.tvmf_query_chunk_size, "checkpoint_chunks": config.tvmf_checkpoint_chunks,
    }


class V4Qwen3Model(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        for index in config.tvmf_layer_indices:
            attention = self.layers[index].self_attn
            attention.tvmf_parameters = TvmfHeadParameters(
                config.num_attention_heads, config.tvmf_kappa, config.tvmf_head_scale,
                config.tvmf_learnable_kappa, config.tvmf_learnable_scale,
                device=attention.q_proj.weight.device,
            )

    def _init_weights(self, module):
        # Transformers dispatches missing-weight initialization to the owning
        # PreTrainedModel (this decoder), not necessarily the outer causal LM.
        if isinstance(module, TvmfHeadParameters):
            module.reset_parameters()
        else:
            super()._init_weights(module)


class V4Qwen3ForCausalLM(Qwen3ForCausalLM):
    _keep_in_fp32_modules = ["tvmf_parameters"]
    _keep_in_fp32_modules_strict = ["tvmf_parameters"]

    def __init__(self, config):
        if getattr(config, "tvmf_version", None) != 4:
            raise ValueError("Construct v4_config first, or load a saved v4 checkpoint")
        register_backend()
        config._attn_implementation = BACKEND_NAME
        # Same construction as Qwen3ForCausalLM, with a decoder that knows how
        # to initialize missing head scalars. Never allocate a second 8B model.
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = V4Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def configure_trainable_parameters(model, strategy="last_block", *, tvmf_layers="last", train_final_norm=False):
    model.requires_grad_(False)
    selected = resolve_layer_indices(tvmf_layers, len(model.model.layers))
    if list(selected) != model.config.tvmf_layer_indices:
        raise ValueError("Training layer selection differs from configured t-vMF layers")
    if strategy == "last_attention":
        model.model.layers[-1].self_attn.requires_grad_(True)
    elif strategy == "last_block":
        model.model.layers[-1].requires_grad_(True)
    elif strategy == "tvmf_blocks":
        for index in selected:
            model.model.layers[index].requires_grad_(True)
    elif strategy == "full":
        model.requires_grad_(True)
    elif strategy != "head_parameters":
        raise ValueError(f"Unknown train strategy: {strategy}")
    # Selected head parameters are always included, including a scalar-only run
    # where Q/K/V have no gradients and chunk recomputation is still necessary.
    for index in selected:
        model.model.layers[index].self_attn.tvmf_parameters.requires_grad_(True)
    if train_final_norm:
        model.model.norm.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError("No trainable parameters: enable kappa/scale or select a block strategy")
    return {"strategy": strategy, "total_parameters": total, "trainable_parameters": trainable,
            "trainable_percent": 100 * trainable / total,
            "head_parameters": sum(p.numel() for n, p in model.named_parameters() if ".tvmf_parameters." in n)}


def assert_strict_last_block(model, *, allow_final_norm=False):
    prefix = f"model.layers.{len(model.model.layers)-1}."
    unexpected = [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith(prefix)
                  and not (allow_final_norm and n.startswith("model.norm."))]
    if unexpected:
        raise RuntimeError(f"Parameters outside the last block are trainable: {unexpected}")


def head_parameter_values(model):
    result = {}
    with torch.no_grad():
        for index in model.config.tvmf_layer_indices:
            kappa, scale = model.model.layers[index].self_attn.tvmf_parameters()
            result[str(index)] = {"kappa": kappa.flatten().cpu().tolist(), "scale": scale.flatten().cpu().tolist()}
    return result


def save_trainable_checkpoint(model, tokenizer, output_dir, metadata):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    names.update(getattr(model.config, "tvmf_inherited_state_keys", []))
    # Include scalar buffers too, so fixed-parameter ablations round trip exactly.
    state = {n: t.detach().cpu().contiguous() for n, t in model.state_dict().items()
             if n in names or ".tvmf_parameters." in n}
    save_file(state, str(output / "trainable.safetensors"), metadata={"format": "pt"})
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    model.config.save_pretrained(output)
    recipe = {**metadata, "format_version": 4, "attention": attention_recipe(model.config),
              "state_keys": sorted(state), "head_values": head_parameter_values(model)}
    (output / CONFIG_FILENAME).write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")


def load_trainable_checkpoint(model, checkpoint_dir):
    path = Path(checkpoint_dir)
    metadata = json.loads((path / CONFIG_FILENAME).read_text())
    state = load_file(str(path / "trainable.safetensors"))
    if metadata.get("format_version") != 4 or not state or set(state) != set(metadata["state_keys"]):
        raise ValueError("Invalid or incomplete v4 compact checkpoint")
    expected_heads = {n for n in model.state_dict() if ".tvmf_parameters." in n}
    if not expected_heads.issubset(state):
        raise ValueError("Compact checkpoint is missing v4 head parameters")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected compact checkpoint keys: {incompatible.unexpected_keys}")
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def load_checkpoint(checkpoint, *, base_model=None, dtype=torch.float32, config=None):
    """Load v4 full/compact tensors without reconstructing scalars from initials."""
    path = Path(checkpoint)
    config = config or AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(config, "tvmf_version", None) != 4:
        raise ValueError("Expected v4 checkpoint; use train --init-checkpoint to import v3 weights")
    register_backend()
    if (path / "trainable.safetensors").is_file():
        metadata = json.loads((path / CONFIG_FILENAME).read_text())
        # The base must be the initialization source, not always vanilla Qwen:
        # frozen blocks can also come from a full v3 checkpoint.
        source = base_model or metadata.get("base_model")
        if not source:
            raise ValueError("Compact checkpoint has no base_model; supply it explicitly")
        model, info = V4Qwen3ForCausalLM.from_pretrained(
            source, config=config, dtype=dtype, local_files_only=True,
            attn_implementation=BACKEND_NAME, output_loading_info=True,
        )
        bad = [n for n in info.get("missing_keys", []) if ".tvmf_parameters." not in n]
        if bad or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise ValueError(f"Base weights do not match v4: {info}")
        load_trainable_checkpoint(model, path)
        model.config.tvmf_inherited_state_keys = metadata["state_keys"]
    else:
        model, info = V4Qwen3ForCausalLM.from_pretrained(
            path, config=config, dtype=dtype, local_files_only=True,
            attn_implementation=BACKEND_NAME, output_loading_info=True,
        )
        if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise ValueError(f"Incomplete or incompatible v4 checkpoint: {info}")
    return model


# Public names for the standalone LYRA release. The original names remain as
# compatibility aliases for checkpoints produced by the research code.
LyraQwen3Model = V4Qwen3Model
LyraQwen3ForCausalLM = V4Qwen3ForCausalLM
lyra_config = v4_config
