"""CPU integration tests for gradients, GQA, cache, precision and checkpoint I/O."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from lyra.data import CausalLMCollator
from lyra.eval_longbench import yarn_config, main as evaluate
from lyra.score_longbench import score_run
from lyra.modeling import (
    CONFIG_FILENAME, V4Qwen3ForCausalLM, assert_strict_last_block,
    configure_trainable_parameters, head_parameter_values, load_checkpoint,
    save_trainable_checkpoint, v4_config,
)
from lyra.train import SupervisedLogitsTrainer, initialize_model, parse_args
from lyra.tvmf_attention import TvmfHeadParameters, tvmf_similarity


class V4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = Qwen3Config(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=128, tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        self.ids = torch.randint(1, 64, (2, 11))

    def model(self, **kwargs):
        return V4Qwen3ForCausalLM(v4_config(self.config, query_chunk_size=3, **kwargs))

    def test_calibrated_identity_zero_slope_and_autograd(self):
        c = torch.linspace(-1, 1, 17, dtype=torch.float64)
        for k in (0.0, 0.1, 4.0, 16.0):
            raw = tvmf_similarity(c, k, "raw")
            expected = (raw + k / (1 + k)) / ((1 + 2 * k) / (1 + k) ** 2)
            torch.testing.assert_close(tvmf_similarity(c, k), expected)
            zero = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            y = tvmf_similarity(zero, k)
            self.assertEqual(y.item(), 0)
            self.assertAlmostEqual(torch.autograd.grad(y, zero)[0].item(), 1)
        torch.testing.assert_close(tvmf_similarity(c, 0.0), c)
        c = torch.rand(1, 4, 2, 3, dtype=torch.float64, requires_grad=True) - 0.5
        k = torch.full((1, 4, 1, 1), 4.0, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(tvmf_similarity, (c, k)))

    def test_query_head_gradients_and_freezing(self):
        for mode in ("raw", "calibrated"):
            model = self.model(similarity_mode=mode)
            stats = configure_trainable_parameters(model)
            assert_strict_last_block(model)
            self.assertEqual(stats["head_parameters"], 8)
            self.assertFalse(hasattr(model.model.layers[0].self_attn, "tvmf_parameters"))
            loss = model(input_ids=self.ids, labels=self.ids).loss
            loss.backward()
            params = model.model.layers[-1].self_attn.tvmf_parameters
            for param in (params.raw_kappa, params.raw_scale):
                self.assertEqual(param.shape, (4,))
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertTrue((param.grad != 0).all())
                self.assertGreater(param.grad.std().item(), 0)
            self.assertIsNone(model.model.layers[0].self_attn.q_proj.weight.grad)
            self.assertIsNotNone(model.model.layers[-1].self_attn.q_proj.weight.grad)

    def test_zero_kappa_matches_native_attention_with_padding(self):
        native = Qwen3ForCausalLM(copy.deepcopy(self.config)).eval()
        mask = torch.ones_like(self.ids)
        mask[1, 7:] = 0
        for mode in ("raw", "calibrated"):
            model = self.model(kappa=0, learnable_kappa=False, learnable_scale=False, similarity_mode=mode).eval()
            model.load_state_dict(native.state_dict(), strict=False)
            with torch.no_grad():
                expected = native(self.ids, attention_mask=mask).logits
                actual = model(self.ids, attention_mask=mask).logits
            torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=1e-5, rtol=1e-4)

    def test_checkpointed_chunks_match_full_forward_and_gradients(self):
        for strategy in ("last_block", "head_parameters"):
            chunked = self.model()
            whole = copy.deepcopy(chunked)
            whole.config.tvmf_query_chunk_size = 100
            whole.config.tvmf_checkpoint_chunks = False
            configure_trainable_parameters(chunked, strategy)
            configure_trainable_parameters(whole, strategy)
            a = chunked(input_ids=self.ids, labels=self.ids).loss
            b = whole(input_ids=self.ids, labels=self.ids).loss
            a.backward()
            b.backward()
            torch.testing.assert_close(a, b)
            for (name, p), (_, q) in zip(chunked.named_parameters(), whole.named_parameters()):
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)

    def test_causality_and_cache_at_native_and_yarn_positions(self):
        for mode in ("raw", "calibrated"):
            for yarn in (False, True):
                config = yarn_config(self.config, 2) if yarn else self.config
                model = V4Qwen3ForCausalLM(v4_config(config, similarity_mode=mode, query_chunk_size=3)).eval()
                params = model.model.layers[-1].self_attn.tvmf_parameters
                with torch.no_grad():
                    params.raw_kappa.add_(torch.arange(4) * 0.3)
                    params.raw_scale.add_(torch.arange(4) * 0.1)
                pos = torch.arange(11).unsqueeze(0).expand(2, -1) + (50000 if yarn else 0)
                changed = self.ids.clone()
                changed[:, 6:] = (changed[:, 6:] + 7) % 64
                with torch.inference_mode():
                    whole = model(self.ids, position_ids=pos, use_cache=False).logits
                    other = model(changed, position_ids=pos, use_cache=False).logits
                    prefill = model(self.ids[:, :6], position_ids=pos[:, :6], use_cache=True)
                    tail = model(self.ids[:, 6:], position_ids=pos[:, 6:], past_key_values=prefill.past_key_values).logits
                    single_cache = None
                    tokens = []
                    for i in range(11):
                        result = model(self.ids[:, i:i+1], position_ids=pos[:, i:i+1],
                                       past_key_values=single_cache, use_cache=True)
                        single_cache = result.past_key_values
                        tokens.append(result.logits)
                torch.testing.assert_close(whole[:, :6], other[:, :6], atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(whole[:, 6:], tail, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(whole, torch.cat(tokens, dim=1), atol=1e-5, rtol=1e-4)

    def test_fp32_scalars_survive_model_cast_and_full_checkpoint(self):
        model = self.model().eval()
        params = model.model.layers[-1].self_attn.tvmf_parameters
        with torch.no_grad():
            params.raw_kappa.add_(torch.tensor([0.00013, 0.17, 0.0035, 1.17]))
            params.raw_scale.add_(torch.tensor([0.13, 0.0007, -0.0345, 0.89]))
        expected = copy.deepcopy(params.state_dict())
        model.bfloat16()
        self.assertEqual(model.model.layers[-1].self_attn.q_proj.weight.dtype, torch.bfloat16)
        for name, value in params.state_dict().items():
            self.assertEqual(value.dtype, torch.float32)
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            loaded = load_checkpoint(temp, dtype=torch.bfloat16).eval()
            for name, value in loaded.model.layers[-1].self_attn.tvmf_parameters.state_dict().items():
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            # A direct .bfloat16() also rounds native nonpersistent RoPE buffers;
            # from_pretrained reconstructs them in FP32. Compare the real loader
            # round trip separately from the explicit parameter-casting check.
            loaded.save_pretrained(temp)
            again = load_checkpoint(temp, dtype=torch.bfloat16).eval()
            with torch.inference_mode():
                torch.testing.assert_close(loaded(self.ids).logits, again(self.ids).logits, rtol=0, atol=0)

    def test_longbench_generation_scoring_and_resume(self):
        vocab = {word: i for i, word in enumerate(["[UNK]", "[PAD]", "[EOS]", "The", "correct", "answer", "is", "(", ")", "A", "B", "C", "D"])}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
        model = self.model().eval()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            data = root / "data.json"
            data.write_text(json.dumps([{"_id": "test-1", "context": "document " * 100, "question": "Choose A",
                                         "choice_A": "A", "choice_B": "B", "choice_C": "C", "choice_D": "D",
                                         "answer": "A", "domain": "Single-Document QA", "sub_domain": "test",
                                         "length": "short", "difficulty": "easy"}]))
            argv = ["--checkpoint", str(checkpoint), "--longbench-v2", str(data), "--device", "cpu",
                    "--max-context-tokens", "96", "--v2-max-new-tokens", "32", "--output-dir", str(root / "results")]
            with patch("lyra.eval_longbench.validate_qwen3_8b"):
                evaluate(argv)
                prediction = next((root / "results").rglob("longbench_v2.jsonl"))
                original = prediction.read_bytes()
                evaluate(argv)
                self.assertEqual(original, prediction.read_bytes())
            row = json.loads(original)
            self.assertIn(row["pred"], "ABCD")
            self.assertTrue(row["input_truncated"])
            self.assertFalse(row["output_truncated"])
            result = score_run(prediction.parent, compensate_unparsed=False)
            self.assertEqual(result["longbench_v2"]["samples_scored"], 1)
            self.assertEqual(result["longbench_v2"]["unparsed_answers"], 0)

    def test_import_v3_and_compact_roundtrip_preserves_frozen_weights(self):
        for compact_v3 in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                base, v3, v4 = root / "base", root / "v3", root / "v4"
                native = Qwen3ForCausalLM(copy.deepcopy(self.config))
                native.save_pretrained(base)
                key = "model.layers.1.self_attn.q_proj.weight"
                with torch.no_grad():
                    native.model.layers[-1].self_attn.q_proj.weight.add_(0.0123)
                if compact_v3:
                    v3.mkdir()
                    save_file({key: native.state_dict()[key]}, str(v3 / "trainable.safetensors"))
                    (v3 / "tvmf_v3_config.json").write_text(json.dumps({"format_version": 3}))
                else:
                    native.config.tvmf_layer_indices = [1]
                    native.config.tvmf_kappa = 4.0
                    native.save_pretrained(v3)
                args = parse_args(["--model", str(base), "--init-checkpoint", str(v3), "--train-file", "unused"])
                model, base_path, _ = initialize_model(args, torch.float32)
                params = model.model.layers[-1].self_attn.tvmf_parameters
                torch.testing.assert_close(params()[0].flatten(), torch.full((4,), 4.0))
                torch.testing.assert_close(params()[1].flatten(), torch.ones(4))
                torch.testing.assert_close(model.state_dict()[key], native.state_dict()[key], rtol=0, atol=0)
                configure_trainable_parameters(model, "head_parameters")
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
                model(input_ids=self.ids, labels=self.ids).loss.backward()
                optimizer.step()
                model.eval()
                save_trainable_checkpoint(model, None, v4, {"base_model": base_path})
                loaded = load_checkpoint(v4, base_model=base_path).eval()
                with torch.inference_mode():
                    torch.testing.assert_close(model(self.ids).logits, loaded(self.ids).logits, rtol=0, atol=0)
                self.assertEqual(head_parameter_values(model), head_parameter_values(loaded))
                # Re-saving after freezing inherited weights must keep those weights.
                configure_trainable_parameters(loaded, "head_parameters")
                save_trainable_checkpoint(loaded, None, root / "v4-again", {"base_model": base_path})
                again = load_checkpoint(root / "v4-again", base_model=base_path).eval()
                torch.testing.assert_close(again.state_dict()[key], native.state_dict()[key], rtol=0, atol=0)

    def test_incomplete_checkpoint_rejected(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            model.save_pretrained(path)
            state = load_file(str(path / "model.safetensors"))
            state.pop("model.layers.1.self_attn.tvmf_parameters.raw_kappa")
            save_file(state, str(path / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_checkpoint(path)

    def test_trainer_updates_scalars_and_resumes_optimizer(self):
        model = self.model()
        configure_trainable_parameters(model)
        dataset = [{"input_ids": row.tolist(), "labels": [-100] * 5 + row[5:].tolist()}
                   for row in self.ids]
        collator = CausalLMCollator(0, pad_to_multiple_of=None)
        with tempfile.TemporaryDirectory() as temp:
            args = TrainingArguments(output_dir=temp, max_steps=2, per_device_train_batch_size=1,
                                     learning_rate=0.001, weight_decay=0.1, save_strategy="steps", save_steps=1,
                                     logging_steps=1, report_to="none", use_cpu=True, remove_unused_columns=False,
                                     dataloader_pin_memory=False, disable_tqdm=True)
            trainer = SupervisedLogitsTrainer(model=model, args=args, train_dataset=dataset,
                                               data_collator=collator, head_learning_rate=0.002)
            original = head_parameter_values(model)
            optimizer = trainer.create_optimizer()
            heads = {id(p) for n, p in model.named_parameters() if ".tvmf_parameters." in n}
            for group in optimizer.param_groups:
                if any(id(p) in heads for p in group["params"]):
                    self.assertEqual(group["lr"], 0.002)
                    self.assertEqual(group["weight_decay"], 0)
            # Response-only selected-logit loss agrees with ordinary full logits.
            batch = collator(dataset)
            torch.testing.assert_close(trainer.compute_loss(model, batch), model(**batch).loss)
            trainer.train()
            self.assertNotEqual(head_parameter_values(model), original)
            self.assertTrue(any("tvmf/layer_1/kappa_mean" in row for row in trainer.state.log_history))
            checkpoint = Path(temp) / "checkpoint-1"
            loaded = load_checkpoint(checkpoint)
            configure_trainable_parameters(loaded)
            resumed = SupervisedLogitsTrainer(model=loaded, args=args, train_dataset=dataset,
                                               data_collator=collator, head_learning_rate=0.002)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 2)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name], atol=1e-7, rtol=1e-5)

    def test_invalid_scalar_initializations(self):
        for k in (-1, 0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TvmfHeadParameters(4, kappa=k)
        for scale in (-1, 0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TvmfHeadParameters(4, head_scale=scale)
        self.assertEqual(parse_args(["--train-file", "unused"]).similarity_mode, "calibrated")


if __name__ == "__main__":
    unittest.main()
