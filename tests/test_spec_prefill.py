# Copyright © 2026 mlx-lm contributors.

import math
import os
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.spec_prefill import (
    SpecPrefillConfig,
    _aggregate_importance,
    _aggregate_pool_then_max,
    _apply_compact_with_tail,
    _avg_pool1d_axis,
    _compute_per_layer_importance,
    _layer_softmax_probs,
    _rope_at_positions,
    _rope_period_table,
    _select_top_blocks,
    _smooth_avg_pool,
    apply_spec_prefill,
    compute_keep_indices,
)


# ---------------------------------------------------------------------------
# Unit tests that don't require a real model
# ---------------------------------------------------------------------------


class TestSmoothing(unittest.TestCase):
    def test_kernel_one_is_identity(self):
        x = mx.array([1.0, 2.0, 3.0, 4.0])
        y = _smooth_avg_pool(x, 1)
        self.assertTrue(mx.array_equal(x, y))

    def test_even_kernel_rejected(self):
        x = mx.arange(10).astype(mx.float32)
        with self.assertRaises(ValueError):
            _smooth_avg_pool(x, 4)

    def test_average_pool_shape_preserved(self):
        x = mx.arange(16).astype(mx.float32)
        y = _smooth_avg_pool(x, 3)
        self.assertEqual(y.shape, x.shape)

    def test_constant_input_is_constant(self):
        x = mx.full((20,), 1.5, dtype=mx.float32)
        y = _smooth_avg_pool(x, 5)
        self.assertTrue(mx.allclose(y, x, atol=1e-6).item())


class TestSelection(unittest.TestCase):
    def test_keep_fraction_one_returns_all(self):
        importance = mx.arange(100).astype(mx.float32)
        idx = _select_top_blocks(importance, 100, block_size=16, keep_fraction=1.0)
        self.assertEqual(idx.size, 100)
        self.assertTrue(mx.array_equal(idx, mx.arange(100, dtype=mx.int32)))

    def test_last_token_always_kept(self):
        # Importance peaks at position 5; last token (99) is uninteresting.
        importance = mx.zeros((100,), dtype=mx.float32)
        importance[5] = 10.0
        idx = _select_top_blocks(importance, 100, block_size=16, keep_fraction=0.1)
        self.assertIn(99, idx.tolist())

    def test_indices_sorted(self):
        # Importance with three peaks at scattered positions.
        importance = mx.zeros((100,), dtype=mx.float32)
        importance[80] = 5.0
        importance[10] = 5.0
        importance[40] = 5.0
        idx = _select_top_blocks(importance, 100, block_size=16, keep_fraction=0.2)
        idx_list = idx.tolist()
        self.assertEqual(idx_list, sorted(idx_list))

    def test_pad_does_not_get_selected(self):
        # M = 17 (one over a 16-block); only the partial block has high score.
        importance = mx.zeros((17,), dtype=mx.float32)
        importance[16] = 1.0
        idx = _select_top_blocks(importance, 17, block_size=16, keep_fraction=0.5)
        # Padded positions (>= 17) must never appear.
        self.assertTrue(all(i < 17 for i in idx.tolist()))

    def test_sink_indices_always_present(self):
        # All importance peaks at the end; block selection would pick the
        # last block, leaving everything < 16 unselected. sink_size=4
        # guarantees [0, 1, 2, 3] survive.
        importance = mx.zeros((64,), dtype=mx.float32)
        importance[60:64] = mx.array([1.0, 2.0, 3.0, 4.0])
        idx = _select_top_blocks(
            importance, 64, block_size=16, keep_fraction=0.1, sink_size=4
        )
        for i in range(4):
            self.assertIn(i, idx.tolist())

    def test_tail_keep_always_present(self):
        # Importance peaks early; without tail_keep the end would be dropped.
        importance = mx.zeros((64,), dtype=mx.float32)
        importance[0:4] = mx.array([4.0, 3.0, 2.0, 1.0])
        idx = _select_top_blocks(
            importance, 64, block_size=16, keep_fraction=0.1, tail_keep=8
        )
        for i in range(56, 64):
            self.assertIn(i, idx.tolist())

    def test_sink_and_tail_compose(self):
        importance = mx.full((96,), 0.1, dtype=mx.float32)
        importance[40:56] = 5.0  # block 2..3 high
        idx = _select_top_blocks(
            importance,
            96,
            block_size=16,
            keep_fraction=0.2,
            sink_size=8,
            tail_keep=8,
        ).tolist()
        # Sink and tail both protected.
        for i in range(8):
            self.assertIn(i, idx)
        for i in range(88, 96):
            self.assertIn(i, idx)
        # The high-importance block must also be present.
        self.assertTrue(any(40 <= i < 56 for i in idx))

    def test_keep_fraction_zero_preserves_protected(self):
        importance = mx.zeros((32,), dtype=mx.float32)
        idx = _select_top_blocks(
            importance,
            32,
            block_size=8,
            keep_fraction=0.0,
            sink_size=4,
            tail_keep=4,
        ).tolist()
        for i in list(range(4)) + list(range(28, 32)):
            self.assertIn(i, idx)


class TestAggregation(unittest.TestCase):
    def test_max_over_layers_mean_over_steps(self):
        # Build [L=2, N=3, M=4]. Layer 0 dominates at m=1, layer 1 at m=2.
        L, N, M = 2, 3, 4
        x = mx.zeros((L, N, M), dtype=mx.float32)
        x[0, :, 1] = mx.array([1.0, 2.0, 3.0])
        x[1, :, 2] = mx.array([10.0, 20.0, 30.0])
        agg = _aggregate_importance(x)
        # m=0: max(0,0)=0 mean over n -> 0
        # m=1: max(1,0)=1 / max(2,0)=2 / max(3,0)=3 -> mean 2
        # m=2: max(0,10)=10 / max(0,20)=20 / max(0,30)=30 -> mean 20
        # m=3: 0
        expected = mx.array([0.0, 2.0, 20.0, 0.0])
        self.assertTrue(mx.allclose(agg, expected, atol=1e-5).item())


class TestAxisPool(unittest.TestCase):
    def test_matches_1d_pool_along_last_axis(self):
        x = mx.random.uniform(shape=(3, 5, 32))
        out = _avg_pool1d_axis(x, 7, axis=-1)
        # Compare slice-by-slice against the [M]-only avg pool.
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ref = _smooth_avg_pool(x[i, j], 7)
                self.assertTrue(mx.allclose(out[i, j], ref, atol=1e-5).item())

    def test_kernel_one_identity_nd(self):
        x = mx.random.uniform(shape=(2, 4, 8))
        out = _avg_pool1d_axis(x, 1, axis=-1)
        self.assertTrue(mx.array_equal(x, out))


class TestPoolThenMax(unittest.TestCase):
    def test_orderings_diverge_when_spikes_share_a_window(self):
        """The two aggregations are mathematically equal when at most one
        (h, n) channel contributes a spike inside any given pool window
        (the max and the windowed-average commute on a single non-zero
        sample). They diverge when two channels each spike INSIDE the
        same window: max_then_pool sums both spike magnitudes (max picks
        each at its own position, then averaging mixes them); pool_then_max
        attenuates each spike by the window size first, then picks the
        single largest – missing the additive contribution.
        """
        H, N, M, kernel = 4, 1, 32, 5
        probs = mx.full((H, N, M), 1.0 / M, dtype=mx.float32)
        probs[0, :, 10] = 0.5
        probs[1, :, 12] = 0.7
        importance_ptm = _aggregate_pool_then_max([probs], kernel)
        per_head_max = probs.max(axis=0)
        importance_mtp = _smooth_avg_pool(per_head_max.mean(axis=0), kernel)
        # Position 11 falls inside [9..13], the window covering both spikes.
        self.assertGreater(
            float(importance_mtp[11].item()), float(importance_ptm[11].item())
        )
        # And at the spike-free side of the array they agree.
        self.assertTrue(
            mx.allclose(importance_ptm[24:], importance_mtp[24:], atol=1e-6).item()
        )

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            _aggregate_pool_then_max([], 3)


class TestPerLayerImportance(unittest.TestCase):
    def test_outlier_key_dominates(self):
        # One key designed to score highest for the query.
        head_dim = 8
        N, M = 4, 12
        # Queries: all equal to [1, 0, ..., 0]
        q_vec = mx.zeros((head_dim,), dtype=mx.float32)
        q_vec[0] = 1.0
        queries = mx.broadcast_to(q_vec, (1, 1, N, head_dim))
        # Keys: random small, but key at position 7 aligned with the query.
        keys = mx.random.normal((1, 1, M, head_dim)) * 0.01
        keys[..., 7, :] = q_vec * 5.0
        per_layer = _compute_per_layer_importance(queries, keys, head_dim)
        self.assertEqual(per_layer.shape, (N, M))
        # The argmax along M should be 7 for every step.
        argmax = per_layer.argmax(axis=-1)
        for n in range(N):
            self.assertEqual(int(argmax[n].item()), 7)


class TestRopeAtPositions(unittest.TestCase):
    def test_matches_standard_rope_at_contiguous_positions(self):
        head_dim = 16
        L = 5
        rope = nn.RoPE(head_dim, traditional=False, base=10000.0)
        x = mx.random.normal((1, 2, L, head_dim))
        # Standard call with offset 3 → positions [3..7]
        std = rope(x, offset=3)
        ours = _rope_at_positions(x, mx.array([3, 4, 5, 6, 7]), rope)
        self.assertTrue(mx.allclose(std, ours, atol=1e-4).item())

    def test_traditional_rope_matches(self):
        head_dim = 16
        L = 4
        rope = nn.RoPE(head_dim, traditional=True, base=10000.0)
        x = mx.random.normal((1, 2, L, head_dim))
        std = rope(x, offset=0)
        ours = _rope_at_positions(x, mx.arange(L), rope)
        self.assertTrue(mx.allclose(std, ours, atol=1e-4).item())

    def test_period_table_present_for_stock_rope(self):
        rope = nn.RoPE(32, traditional=False, base=10000.0)
        period, rotated, traditional, scale = _rope_period_table(rope)
        self.assertEqual(rotated, 32)
        self.assertFalse(traditional)
        self.assertEqual(period.size, 16)


# ---------------------------------------------------------------------------
# End-to-end integration tests against a tiny model (skipped if env unset)
# ---------------------------------------------------------------------------


SMOKE_MODEL_PATH = os.environ.get(
    "MLX_LM_SPEC_PREFILL_TEST_MODEL",
    "/Users/Shared/models/qwen3.5-0.8b-mlx-4bit",
)


@unittest.skipUnless(
    os.path.exists(SMOKE_MODEL_PATH),
    f"Smoke test model not found at {SMOKE_MODEL_PATH}",
)
class TestSpecPrefillSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load

        cls.model, cls.tokenizer = load(SMOKE_MODEL_PATH)

    def _make_prompt(self, n: int) -> mx.array:
        vocab = self.model.args.text_config.get("vocab_size", 32000) if hasattr(
            self.model, "args"
        ) else 32000
        if isinstance(vocab, dict):
            vocab = 32000
        # Avoid sampling special / very large IDs by limiting to 1..1024.
        rng = mx.random.uniform(low=1, high=1024, shape=(n,))
        return rng.astype(mx.uint32)

    def test_compact_with_tail_runs(self):
        prompt = self._make_prompt(1024)
        cfg = SpecPrefillConfig(
            speculator_threshold=256,
            keep_fraction=0.3,
            block_size=16,
            pool_kernel=13,
            lookahead_steps=4,
            position_layout="compact_with_tail",
            tail_size=128,
            prefill_step_size=512,
        )
        prompt_cache = make_prompt_cache(self.model)
        seed, cleanup = apply_spec_prefill(
            self.model, self.model, prompt, prompt_cache, cfg
        )
        try:
            self.assertEqual(seed.size, 1)
        finally:
            cleanup()

    def test_both_aggregation_modes_produce_valid_selections(self):
        """Both ``pool_then_max`` and ``max_then_pool`` should return a
        non-empty sorted int32 index set that includes the last token."""
        prompt = self._make_prompt(1024)
        for agg in ("pool_then_max", "max_then_pool"):
            idx = compute_keep_indices(
                self.model,
                prompt,
                lookahead_steps=4,
                pool_kernel=13,
                block_size=16,
                keep_fraction=0.3,
                prefill_step_size=512,
                sink_size=8,
                tail_keep=64,
                aggregation=agg,
            )
            idx_list = idx.tolist()
            self.assertGreater(len(idx_list), 0)
            self.assertEqual(idx_list, sorted(idx_list))
            self.assertEqual(idx_list[-1], prompt.size - 1)
            # Sink + tail guarantees apply to both modes.
            for i in range(8):
                self.assertIn(i, idx_list)
            for i in range(prompt.size - 64, prompt.size):
                self.assertIn(i, idx_list)

    def test_long_context_decodes_multiple_tokens(self):
        """Regression for the chunked-prefill / selection class of bugs.

        Drives ``stream_generate`` end-to-end on an 8k prompt – well over
        the ``prefill_step_size=2048`` chunk boundary post-selection –
        and asserts that more than the seed token comes out. Catches
        both the original ``mx.clear_cache()``-during-chunked-prefill
        bug and any future regression where the selection drops
        structurally-important tokens (chat-template footer, attention
        sink) and the model emits an immediate EOS.
        """
        from mlx_lm.generate import stream_generate

        prompt = self._make_prompt(8192)
        tokens_produced = 0
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=32,
            speculator=self.model,
            speculator_threshold=1024,
            keep_fraction=0.3,
            block_size=16,
            pool_kernel=13,
            lookahead_steps=4,
            position_layout="compact_with_tail",
            tail_size=256,
            sink_size=16,
        ):
            tokens_produced = resp.generation_tokens
        # If selection or cache state is broken, this would be 1 (immediate EOS).
        self.assertGreater(tokens_produced, 4)

    def test_32k_context_decodes_multiple_tokens(self):
        """Regression at 32k context against the FINDINGS_v5 knife-edge.

        Both this implementation and the reference (anemll) harness
        exhibit a degenerate-continuation band at
        ``keep_fraction in [0.25, 0.6]`` on Qwen3.6-class models with
        repetitive prompts. The defaults (``keep_fraction=0.2``,
        ``block_size=32``) sit at the empirically robust low-density
        point. This test pins that with the new defaults so future
        changes to selection or layout that would re-cross the knife-edge
        get caught.
        """
        from mlx_lm.generate import stream_generate

        prompt = self._make_prompt(32768)
        tokens_produced = 0
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=64,
            speculator=self.model,
            speculator_threshold=1024,
            # Intentionally use the library defaults (keep_fraction=0.2,
            # block_size=32). Do not specialise them here – the point of
            # this test is that the defaults stay in the safe band.
        ):
            tokens_produced = resp.generation_tokens
        self.assertGreater(tokens_produced, 16)

    def test_original_layout_cleanup_restores_rope(self):
        # Snapshot the rope module identities before / after.
        snap_before = []
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "rope"):
                snap_before.append((attn, attn.rope))

        prompt = self._make_prompt(512)
        cfg = SpecPrefillConfig(
            speculator_threshold=128,
            keep_fraction=0.3,
            block_size=16,
            pool_kernel=13,
            lookahead_steps=4,
            position_layout="original",
            tail_size=64,
            prefill_step_size=512,
        )
        prompt_cache = make_prompt_cache(self.model)
        seed, cleanup = apply_spec_prefill(
            self.model, self.model, prompt, prompt_cache, cfg
        )
        cleanup()
        # Every rope module must be back to the original object.
        for (attn, orig), (attn2, _) in zip(snap_before, snap_before):
            self.assertIs(attn.rope, orig)


if __name__ == "__main__":
    unittest.main()
