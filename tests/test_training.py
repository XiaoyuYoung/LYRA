"""CPU tests for YaRN configuration and long-context training."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM

from lyra.data import CausalLMCollator
from lyra.modeling import V4Qwen3ForCausalLM, v4_config, configure_trainable_parameters, load_checkpoint, save_trainable_checkpoint
from lyra.rope import training_rope_config
from lyra.train import initialize_model, parse_args, make_training_arguments, SupervisedLogitsTrainer




class LongTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=1, head_dim=16,
                                 max_position_embeddings=40960, tie_word_embeddings=False)

    def test_yarn_is_constructed_before_weight_load_and_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            native = Qwen3ForCausalLM(copy.deepcopy(self.config))
            native.save_pretrained(base)
            for limit, factor in ((65536, 2), (131072, 4)):
                args = parse_args(["--model", str(base), "--train-file", "unused", "--max-length", str(limit), "--yarn-factor", str(factor)])
                model, source, _ = initialize_model(args, torch.float32)
                self.assertEqual(model.model.rotary_emb.rope_type, "yarn")
                self.assertEqual(model.config.max_position_embeddings, limit)
                self.assertFalse(torch.equal(native.model.rotary_emb.inv_freq, model.model.rotary_emb.inv_freq))
                configure_trainable_parameters(model)
                ids = torch.randint(0, 64, (1, 7))
                positions = torch.arange(limit - 7, limit).unsqueeze(0)
                loss = model(input_ids=ids, position_ids=positions, labels=ids).loss
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for p in model.model.layers[-1].self_attn.tvmf_parameters.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                model.eval()
                for compact in (False, True):
                    output = root / f"{limit}-{compact}"
                    if compact:
                        save_trainable_checkpoint(model, None, output, {"base_model": source})
                    else:
                        model.save_pretrained(output)
                    loaded = load_checkpoint(output, base_model=source if compact else None).eval()
                    self.assertEqual(loaded.model.rotary_emb.rope_type, "yarn")
                    self.assertEqual(loaded.config.max_position_embeddings, limit)
                    with torch.no_grad():
                        torch.testing.assert_close(model(ids, position_ids=positions).logits,
                                                   loaded(ids, position_ids=positions).logits, rtol=0, atol=0)

    def test_reject_length_and_resume_rope_mismatch(self):
        with self.assertRaises(ValueError):
            training_rope_config(self.config, 65536)
        with self.assertRaises(ValueError):
            training_rope_config(self.config, 131072, 2)
        yarn = training_rope_config(self.config, 65536, 2)
        with self.assertRaises(ValueError):
            training_rope_config(yarn, 131072, 4, resume=True)
        with self.assertRaises(ValueError):
            training_rope_config(yarn, 131072, resume=True)
        self.assertEqual(training_rope_config(yarn, 65536, resume=True).rope_parameters["factor"], 2)

    def test_trainer_non_reentrant_checkpointing_with_frozen_prefix(self):
        model = V4Qwen3ForCausalLM(v4_config(training_rope_config(self.config, 65536, 2), query_chunk_size=3))
        configure_trainable_parameters(model)
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(["--train-file", "unused", "--output-dir", temp, "--gradient-checkpointing"])
            training = make_training_arguments(args, dtype=torch.float32, has_eval=False)
            self.assertEqual(training.gradient_checkpointing_kwargs, {"use_reentrant": False})
            training.max_steps = 1
            training.use_cpu = True
            training.dataloader_pin_memory = False
            training.disable_tqdm = True
            dataset = [{"input_ids": list(range(1, 9)), "labels": [-100] * 4 + list(range(5, 9))}]
            trainer = SupervisedLogitsTrainer(model=model, args=training, train_dataset=dataset,
                                               data_collator=CausalLMCollator(0))
            result = trainer.train()
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))



if __name__ == "__main__":
    unittest.main()
