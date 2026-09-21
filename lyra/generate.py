"""Generate text using learned per-head scalars from a LYRA checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

if __package__:
    from .modeling import CONFIG_FILENAME, load_checkpoint, validate_qwen3_8b
else:
    from modeling import CONFIG_FILENAME, load_checkpoint, validate_qwen3_8b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", help="Optional compact-checkpoint base override.")
    parser.add_argument("--prompt", default="long-context")
    parser.add_argument("--prompt-file", type=Path, help="Read a UTF-8 prompt from a local file.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.query_chunk_size <= 0 or args.max_new_tokens <= 0:
        parser.error("Token and chunk limits must be positive")
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    validate_qwen3_8b(config)
    config.tvmf_query_chunk_size, config.tvmf_checkpoint_chunks = args.query_chunk_size, False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU node or --device cpu")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if args.device.startswith("cuda") else torch.float32
    model = load_checkpoint(args.checkpoint, base_model=args.base_model, dtype=dtype, config=config).to(args.device).eval()
    tokenizer_path = args.checkpoint
    if not (tokenizer_path / "tokenizer_config.json").is_file():
        metadata = json.loads((args.checkpoint / CONFIG_FILENAME).read_text())
        tokenizer_path = args.base_model or metadata["base_model"]
        if not tokenizer_path:
            raise ValueError("Compact checkpoints without tokenizer files require --base-model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(args.device)
    if inputs["input_ids"].shape[1] + args.max_new_tokens > config.max_position_embeddings:
        raise ValueError("Prompt plus generation budget exceeds checkpoint context capacity")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
