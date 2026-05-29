# Copyright © 2026 mlx-lm contributors.
#
# A clean-room implementation of SpecPrefill (Liu, Chen & Zhang, 2025;
# arXiv:2502.02789), built against the mlx-lm model APIs.
#
# The technique uses a small "speculator" model to score prompt-token
# importance, then has the main model prefill only the top-K fraction.
# See ``specprefill_clean_room_spec.md`` for the full spec this implements.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .models import cache as cache_module


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------


@dataclass
class SpecPrefillConfig:
    """Tunable parameters for SpecPrefill.

    The defaults are tuned for production use on Apple Silicon. The
    ``position_layout`` default deviates from a strict reading of the
    SpecPrefill paper – see the field doc below.
    """

    speculator_threshold: int = 8192
    # Top-K fraction of blocks to keep after the per-block mean.
    #
    # **Stability of ``keep_fraction``.** Empirical testing on
    # Qwen3.6-35B-A3B-4bit + Qwen3.5-0.8B-4bit at 32k context showed a
    # knife-edge: values in ``[0.25, 0.6]`` produced a degenerate
    # continuation (immediate EOS), while ``keep_fraction <= 0.2`` and
    # ``keep_fraction ~= 0.8`` were robust. This appears to be a property
    # of SpecPrefill's attention-based selection landing in different
    # cache-state regimes on this particular model + prompt; both this
    # implementation and the reference (anemll harness) implementation
    # exhibit the same shape, so it is not a clean-room defect. Behaviour
    # may vary by model family; users running on a new model should sweep
    # ``keep_fraction`` at their target context length before committing
    # to a value. The default 0.2 was chosen as the empirically robust
    # low-density point.
    keep_fraction: float = 0.2
    # Block size for §3.2.3 selection. The paper text gives "16 or 32 are
    # reasonable defaults"; 32 is chosen here because it lands in the
    # safe band of the keep_fraction knife-edge above on tested workloads
    # (Qwen3.6-35B-A3B-4bit + Qwen3.5-0.8B-4bit at 32k context). 16 also
    # works on prompts not in the failing band; both are paper-faithful.
    block_size: int = 32
    pool_kernel: int = 13
    lookahead_steps: int = 8
    # Number of leading prompt tokens to always keep in the selected
    # set, regardless of block-mean importance. These act as "attention
    # sinks" – a small head of the sequence the model has learnt to
    # rely on for stability. Cheap insurance against block-grouping
    # noise dropping early structurally-important tokens.
    sink_size: int = 16
    # The paper specifies non-contiguous original position IDs for the
    # main model's prefill (``"original"`` here). On Apple Silicon /
    # Qwen3.6-class models this layout decodes ~3× slower than dense
    # due to a scattered-RoPE-phase issue in MLX's attention path; the
    # ``"compact_with_tail"`` extension (history tokens compacted to
    # contiguous positions, last ``tail_size`` tokens kept verbatim)
    # preserves recall in practice while decoding at dense baseline.
    # Default is the extension; pass ``position_layout="original"`` for
    # strict paper fidelity.
    position_layout: str = "compact_with_tail"
    tail_size: int = 256
    prefill_step_size: int = 2048
    # Aggregation order. The paper describes "max over (L, H), mean over
    # N" (§3.2.2) and 1-D average pooling along the sequence axis (§3.2.3)
    # as separate steps but does not pin the *order* of pooling vs the
    # aggregation. Two interpretations:
    #
    #   "pool_then_max" (default) — smooth each (l, h, n) softmax slice
    #     along M first, then max-collapse over (L, H), then mean over N.
    #     Each (l, h) channel's energy is spread spatially before
    #     competing for the max, so a spike in any one head at position p
    #     contributes to a neighbourhood around p in the post-aggregation
    #     importance. Empirically more robust at long contexts (≥ 32k)
    #     where the alternative drops mid-prompt tokens it shouldn't.
    #
    #   "max_then_pool" — paper-faithful reading: max over (L, H), mean
    #     over N first, then smooth the resulting [M] vector. Sharper /
    #     spikier; favors positions with strong single-(l, h) attention
    #     spikes. Suitable for short and mid-length contexts where the
    #     spikes are reliable signal.
    aggregation: str = "pool_then_max"


# ---------------------------------------------------------------------------
# RoPE wrapping helpers
# ---------------------------------------------------------------------------


def _layer_attention_rope(layer: nn.Module):
    """Return ``(attn_module, rope_module)`` for a layer, or ``None``.

    Layers without an attention sub-module with a ``rope`` attribute
    (e.g. linear/SSM-style layers) are skipped.
    """
    attn = getattr(layer, "self_attn", None)
    if attn is None or not hasattr(attn, "rope"):
        return None
    return attn, attn.rope


def _rope_period_table(rope_module) -> Tuple[mx.array, int, bool, float]:
    """Extract the period table, rotated dim count, traditional flag, and
    pre-scale (``mscale`` / ``_scale``) from a RoPE module.

    Falls back to recomputing the period table from ``base``/``dims``/``scale``
    for the stock ``mlx.nn.RoPE`` module which does not store ``_freqs``.
    """

    # Number of rotated dims (variants use ``.dims`` or ``.dim``)
    rotated = getattr(rope_module, "dims", None)
    if rotated is None:
        rotated = getattr(rope_module, "dim", None)
    if rotated is None:
        raise ValueError(
            f"Cannot determine rotated dims for RoPE module {type(rope_module).__name__}"
        )

    traditional = bool(getattr(rope_module, "traditional", False))

    pre_scale = getattr(rope_module, "_scale", None)
    if pre_scale is None:
        pre_scale = getattr(rope_module, "mscale", None)
    if pre_scale is None:
        pre_scale = 1.0
    pre_scale = float(pre_scale)

    period = getattr(rope_module, "_freqs", None)
    if period is None:
        base = float(getattr(rope_module, "base", 10000.0))
        period = base ** (mx.arange(0, rotated, 2, dtype=mx.float32) / rotated)
        rope_scale = getattr(rope_module, "scale", 1.0) or 1.0
        rope_scale = float(rope_scale)
        if rope_scale != 1.0:
            # ``mlx.nn.RoPE``'s ``scale`` divides the position (linear scaling
            # for long context); equivalent to multiplying the period table.
            period = period / rope_scale
    period = period.astype(mx.float32)
    return period, int(rotated), traditional, pre_scale


def _rope_at_positions(
    x: mx.array, positions: mx.array, rope_module
) -> mx.array:
    """Apply RoPE to ``x`` at arbitrary positions ``positions`` (shape ``[L]``).

    Replicates ``rope_module(x, offset=...)`` semantics but allows
    per-token positions instead of a contiguous ``offset + arange(L)``.
    """
    period, rotated, traditional, pre_scale = _rope_period_table(rope_module)

    D = x.shape[-1]
    if rotated < D:
        x_rot = x[..., :rotated]
        x_pass = x[..., rotated:]
    else:
        x_rot = x
        x_pass = None

    if pre_scale != 1.0:
        x_rot = x_rot * pre_scale

    angles = positions[:, None].astype(mx.float32) / period[None, :]
    cos = mx.cos(angles).astype(x.dtype)[None, None, :, :]
    sin = mx.sin(angles).astype(x.dtype)[None, None, :, :]

    half = rotated // 2
    if traditional:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y_rot = mx.stack([y1, y2], axis=-1)
        y_rot = y_rot.reshape(*y1.shape[:-1], 2 * half)
    else:
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y_rot = mx.concatenate([y1, y2], axis=-1)

    if x_pass is not None:
        y_rot = mx.concatenate([y_rot, x_pass], axis=-1)
    return y_rot


# ---------------------------------------------------------------------------
# Query capture via RoPE wrappers
# ---------------------------------------------------------------------------


def _attn_head_counts(attn) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(n_heads, n_kv_heads)`` for an attention module if both
    are introspectable, else ``(None, None)``.

    Covers Llama (``n_heads``/``n_kv_heads``) and Qwen3 / Qwen3-Next /
    Qwen3.5 (``num_attention_heads``/``num_key_value_heads``).
    """
    n_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_attention_heads", None)
    n_kv = getattr(attn, "n_kv_heads", None) or getattr(attn, "num_key_value_heads", None)
    if n_heads is None or n_kv is None:
        return None, None
    return int(n_heads), int(n_kv)


class _QueryCaptureWrapper:
    """Wraps a RoPE module so the *query* call per layer per step is
    recorded for later importance scoring.

    Two strategies, in order of preference:

      * **shape-based** (GQA models, i.e. ``n_heads > n_kv_heads``): the
        wrapper captures only calls whose input has ``shape[1] == n_heads``.
        This is architecture-agnostic – it does not assume Q is rotated
        before K.
      * **parity fallback** (MHA models, where Q and K have identical
        head counts and shape can't distinguish them): the wrapper
        toggles on each call within a step, treating call 0 as Q. This
        relies on every attention implementation in mlx-lm rotating Q
        before K in its ``__call__`` – true at the time of writing for
        Llama, Qwen3, Qwen3-Next, and Qwen3.5.

    ``reset_step()`` is called between forward passes so the parity
    counter starts on the Q branch each step.
    """

    def __init__(self, inner, sink: List[mx.array], n_heads: Optional[int], n_kv: Optional[int]):
        self._inner = inner
        self._sink = sink
        self._n_heads = n_heads
        self._n_kv = n_kv
        self._shape_mode = (
            n_heads is not None and n_kv is not None and n_heads != n_kv
        )
        self._step_calls = 0

    def reset_step(self):
        self._step_calls = 0

    def __call__(self, x, offset=0):
        out = self._inner(x, offset=offset)
        if self._shape_mode:
            if x.shape[1] == self._n_heads:
                self._sink.append(out)
        else:
            if self._step_calls % 2 == 0:
                self._sink.append(out)
        self._step_calls += 1
        return out


def _install_query_capture(
    model: nn.Module,
) -> Tuple[List[List[mx.array]], List[Tuple[Any, Any, _QueryCaptureWrapper]]]:
    """Install ``_QueryCaptureWrapper`` on each attention layer's RoPE.

    Returns ``(captured, handles)``: ``captured[i]`` is the (initially
    empty) list that will collect post-RoPE queries for layer ``i`` across
    lookahead steps. Layers without rotary attention occupy a slot in
    ``captured`` but the slot stays empty.
    """
    captured: List[List[mx.array]] = []
    handles: List[Tuple[Any, Any, _QueryCaptureWrapper]] = []
    for layer in model.layers:
        slot: List[mx.array] = []
        captured.append(slot)
        info = _layer_attention_rope(layer)
        if info is None:
            continue
        attn, rope = info
        n_heads, n_kv = _attn_head_counts(attn)
        wrap = _QueryCaptureWrapper(rope, slot, n_heads, n_kv)
        attn.rope = wrap
        handles.append((attn, rope, wrap))
    return captured, handles


def _remove_query_capture(handles):
    for attn, orig_rope, _ in handles:
        attn.rope = orig_rope


def _reset_capture_step(handles):
    for _, _, wrap in handles:
        wrap.reset_step()


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


def _layer_softmax_probs(
    queries: mx.array, keys: mx.array, head_dim: int
) -> mx.array:
    """Compute per-head softmax attention probabilities for one layer.

    Args:
      queries: ``[1, n_heads, N, head_dim]`` – captured lookahead queries.
      keys: ``[1, n_kv_heads, M, head_dim]`` – speculator cache keys for
        the original ``M`` prompt positions (lookahead positions excluded).
      head_dim: per-head dimension.

    Returns:
      ``[n_heads, N, M]`` float32 attention probabilities (softmax over M).
    """
    n_heads = queries.shape[1]
    n_kv = keys.shape[1]
    if n_heads != n_kv:
        keys = mx.repeat(keys, n_heads // n_kv, axis=1)
    scale = head_dim ** -0.5
    q = queries.astype(mx.float32)
    k = keys.astype(mx.float32)
    scores = (q @ k.swapaxes(-2, -1)) * scale
    probs = mx.softmax(scores, axis=-1)
    return probs[0]  # [n_heads, N, M]


def _compute_per_layer_importance(
    queries: mx.array, keys: mx.array, head_dim: int
) -> mx.array:
    """Backwards-compatible wrapper that returns ``[N, M]`` post-head-max
    probabilities for one layer. Used only by the ``"max_then_pool"``
    aggregation path.
    """
    return _layer_softmax_probs(queries, keys, head_dim).max(axis=0)


def _stack_layer_importances(per_layer: List[mx.array]) -> mx.array:
    """Stack per-layer ``[N, M]`` scores into ``[L, N, M]``.

    All layers must share ``(N, M)``. Layers contributing nothing (e.g.
    linear-attention layers without RoPE) are filtered out.
    """
    if not per_layer:
        raise ValueError(
            "SpecPrefill scoring produced no attention contributions; "
            "is the speculator a pure-attention transformer?"
        )
    return mx.stack(per_layer, axis=0)


def _aggregate_importance(stacked: mx.array) -> mx.array:
    """Implements the paper-faithful order: ``mean_n( max_{l,h}( probs ) )``.

    Heads were already maxed inside ``_compute_per_layer_importance``;
    here we max over layers and mean over lookahead steps.

    Args:
      stacked: ``[L, N, M]`` per-layer post-head-max scores.

    Returns:
      ``[M]`` importance vector (float32).
    """
    per_step = stacked.max(axis=0)  # [N, M]
    return per_step.mean(axis=0)  # [M]


def _avg_pool1d_axis(x: mx.array, kernel: int, axis: int) -> mx.array:
    """Symmetric 1-D average pooling along ``axis`` with edge padding.

    Generalises :func:`_smooth_avg_pool` to N-D inputs without flattening
    the other dimensions. Implementation uses ``cumsum`` along ``axis``
    after edge padding, which is the fastest single-pass way to compute
    a moving average and avoids allocating an explicit windowed view.
    """
    if kernel <= 1:
        return x
    if kernel % 2 == 0:
        raise ValueError(f"pool_kernel must be odd, got {kernel}")
    half = (kernel - 1) // 2
    last = x.shape[axis]
    take_left = mx.take(x, mx.array([0], dtype=mx.int32), axis=axis)
    take_right = mx.take(x, mx.array([last - 1], dtype=mx.int32), axis=axis)
    left = mx.repeat(take_left, half, axis=axis)
    right = mx.repeat(take_right, half, axis=axis)
    padded = mx.concatenate([left, x, right], axis=axis).astype(mx.float32)
    cum = mx.cumsum(padded, axis=axis)
    # zero-prefix along `axis` so cum[k:] - cum[:-k] gives the windowed sum
    zero_shape = list(cum.shape)
    zero_shape[axis] = 1
    zero = mx.zeros(zero_shape, dtype=cum.dtype)
    cum = mx.concatenate([zero, cum], axis=axis)
    # slice both halves along `axis`
    end = cum.shape[axis]
    upper = _slice_axis(cum, axis, kernel, end)
    lower = _slice_axis(cum, axis, 0, end - kernel)
    return (upper - lower) / kernel


def _slice_axis(x: mx.array, axis: int, start: int, stop: int) -> mx.array:
    """Slice ``x`` along ``axis`` from ``start`` to ``stop``."""
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(start, stop)
    return x[tuple(sl)]


def _aggregate_pool_then_max(
    per_layer_probs: List[mx.array], pool_kernel: int
) -> mx.array:
    """Pool first, then max-collapse (L, H), then mean over N.

    Args:
      per_layer_probs: list of ``[n_heads, N, M]`` float32 softmax
        outputs, one per (rotary) layer. ``n_heads`` may differ per
        layer (GQA repeat already applied upstream).
      pool_kernel: 1-D average-pool kernel applied along the M axis.

    Returns:
      ``[M]`` importance vector (float32). Caller does *not* need to
      smooth this – the smoothing already happened pre-max.
    """
    if not per_layer_probs:
        raise ValueError(
            "SpecPrefill scoring produced no attention contributions; "
            "is the speculator a pure-attention transformer?"
        )
    # Concat all heads from all layers along the leading axis →
    # [L*H_total, N, M]. Different layers may contribute different
    # numbers of heads, so we concat rather than stack.
    combined = mx.concatenate(per_layer_probs, axis=0)
    if pool_kernel > 1:
        combined = _avg_pool1d_axis(combined, pool_kernel, axis=-1)
    max_scores = combined.max(axis=0)  # [N, M]
    return max_scores.mean(axis=0)  # [M]


# ---------------------------------------------------------------------------
# Smoothing + selection
# ---------------------------------------------------------------------------


def _smooth_avg_pool(x: mx.array, kernel: int) -> mx.array:
    """Symmetric 1-D average pooling with edge padding.

    Args:
      x: ``[M]`` float array.
      kernel: odd window size.

    Returns:
      ``[M]`` smoothed array.
    """
    if kernel <= 1:
        return x
    if kernel % 2 == 0:
        raise ValueError(f"pool_kernel must be odd, got {kernel}")
    half = (kernel - 1) // 2
    left = mx.repeat(x[:1], half)
    right = mx.repeat(x[-1:], half)
    padded = mx.concatenate([left, x, right]).astype(mx.float32)
    cum = mx.concatenate([mx.zeros((1,), padded.dtype), mx.cumsum(padded)])
    return (cum[kernel:] - cum[:-kernel]) / kernel


def _select_top_blocks(
    importance: mx.array,
    prompt_len: int,
    block_size: int,
    keep_fraction: float,
    sink_size: int = 0,
    tail_keep: int = 0,
) -> mx.array:
    """Reshape into blocks, pick top-K by mean, return sorted flat indices.

    Args:
      importance: ``[M]`` smoothed per-token importance.
      prompt_len: ``M``.
      block_size: chunking size for §3.2.3 selection.
      keep_fraction: top fraction of blocks to keep.
      sink_size: always-keep ``[0, sink_size)`` (attention sink). The
        block-mean selection is unaware of this set – the sink indices
        are unioned in afterwards. Defaults to 0 (off).
      tail_keep: always-keep ``[M - tail_keep, M)``. Same union rule as
        ``sink_size``. Important for ``position_layout="original"`` to
        guarantee the chat-template footer survives selection (the
        ``"compact_with_tail"`` layout protects the tail independently
        but passing it here is harmless – the indices are already in
        the keep set, no duplication occurs). Defaults to 0 (off).

    Edge cases:
      * ``keep_fraction >= 1.0`` returns ``arange(M)`` unchanged.
      * The last prompt token (index ``M-1``) is always included so the
        existing ``_step`` seed-token logic in ``generate_step`` remains
        correct.
    """
    if keep_fraction >= 1.0:
        return mx.arange(prompt_len, dtype=mx.int32)
    if keep_fraction <= 0.0:
        # Degenerate: keep at least the protected ranges + the last token.
        protected = set()
        if sink_size > 0:
            protected.update(range(min(sink_size, prompt_len)))
        if tail_keep > 0:
            protected.update(range(max(0, prompt_len - tail_keep), prompt_len))
        protected.add(prompt_len - 1)
        return mx.array(sorted(protected), dtype=mx.int32)

    num_blocks = (prompt_len + block_size - 1) // block_size
    pad = num_blocks * block_size - prompt_len
    if pad > 0:
        importance = mx.concatenate(
            [importance.astype(mx.float32), mx.full((pad,), -1e30, dtype=mx.float32)]
        )
    blocks = importance.reshape(num_blocks, block_size)
    block_means = blocks.mean(axis=1)

    k = max(1, int(math.ceil(keep_fraction * num_blocks)))
    k = min(k, num_blocks)
    order = mx.argsort(-block_means)
    top = order[:k]
    top_list = sorted(int(b) for b in top.tolist())

    selected: List[int] = []
    for b in top_list:
        start = b * block_size
        end = min(start + block_size, prompt_len)
        selected.extend(range(start, end))

    selected_set = set(selected)
    if sink_size > 0:
        selected_set.update(range(min(sink_size, prompt_len)))
    if tail_keep > 0:
        selected_set.update(range(max(0, prompt_len - tail_keep), prompt_len))
    selected_set.add(prompt_len - 1)
    return mx.array(sorted(selected_set), dtype=mx.int32)


# ---------------------------------------------------------------------------
# Speculator scoring
# ---------------------------------------------------------------------------


def _chunked_prefill(
    model: nn.Module, prompt: mx.array, prompt_cache: List[Any], step: int
):
    """Prefill ``model`` on ``prompt`` in chunks of size ``step``.

    Returns the logits from the LAST chunk (so the caller can sample the
    first follow-on token).

    Note: ``mx.clear_cache()`` is deliberately NOT called between chunks.
    Doing so during a multi-chunk prefill that immediately precedes a
    sample step caused stale cache reads against Qwen3.6-class models at
    prompts ≥ ~6.5k post-selection tokens (the cache appeared correct
    after ``mx.eval`` but the next forward sampled essentially noise –
    typically a single EOS token). One ``mx.clear_cache()`` at the end is
    sufficient for memory hygiene.
    """
    M = prompt.size
    i = 0
    last_logits = None
    while i < M:
        n = min(step, M - i)
        chunk = prompt[i : i + n][None]
        last_logits = model(chunk, cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        i += n
    mx.clear_cache()
    return last_logits


def _speculator_keys_for_prompt(
    prompt_cache: List[Any], prompt_len: int
) -> List[Optional[mx.array]]:
    """For each layer in the speculator cache, return ``cache.keys[..., :M, :]``
    if the layer holds per-token keys (a standard KV cache), or ``None``
    if it doesn't (e.g. linear/SSM-style caches).
    """
    out: List[Optional[mx.array]] = []
    for c in prompt_cache:
        keys = getattr(c, "keys", None)
        if keys is None or not isinstance(keys, mx.array):
            out.append(None)
            continue
        # KVCache stores keys as [B, n_kv_heads, max_offset, head_dim]
        if keys.ndim != 4:
            out.append(None)
            continue
        if keys.shape[2] < prompt_len:
            # Sliding-window or otherwise truncated; skip rather than guess.
            out.append(None)
            continue
        out.append(keys[..., :prompt_len, :])
    return out


def compute_keep_indices(
    speculator: nn.Module,
    prompt: mx.array,
    *,
    lookahead_steps: int,
    pool_kernel: int,
    block_size: int,
    keep_fraction: float,
    prefill_step_size: int = 2048,
    sink_size: int = 0,
    tail_keep: int = 0,
    aggregation: str = "pool_then_max",
) -> mx.array:
    """Run the speculator's prefill + lookahead, score importance, and
    return the sorted ``int32`` array of selected prompt indices.

    ``sink_size`` / ``tail_keep`` extend the kept set with always-on
    leading / trailing index ranges; see :func:`_select_top_blocks`.

    ``aggregation`` selects between ``"pool_then_max"`` (default; smooth
    each (l, h, n) softmax slice along M, then max-collapse over (L, H),
    then mean over N) and ``"max_then_pool"`` (paper-faithful: max over
    (L, H), mean over N, then smooth the resulting [M] vector). See the
    ``SpecPrefillConfig.aggregation`` docstring for the rationale.
    """
    if keep_fraction >= 1.0:
        return mx.arange(prompt.size, dtype=mx.int32)

    spec_cache = cache_module.make_prompt_cache(speculator)
    M = int(prompt.size)

    last_logits = _chunked_prefill(speculator, prompt, spec_cache, prefill_step_size)

    captured, handles = _install_query_capture(speculator)
    try:
        # Sample the first lookahead token from the prefill's last logits.
        # We use greedy sampling for the speculator regardless of the user's
        # main-model sampler — importance scoring is deterministic.
        y = mx.argmax(last_logits[:, -1, :], axis=-1).astype(prompt.dtype)
        mx.eval(y)
        for _ in range(lookahead_steps):
            _reset_capture_step(handles)
            logits = speculator(y[None], cache=spec_cache)
            y = mx.argmax(logits[:, -1, :], axis=-1).astype(prompt.dtype)
            mx.eval(y)
    finally:
        _remove_query_capture(handles)

    layer_keys = _speculator_keys_for_prompt(spec_cache, M)

    if aggregation == "pool_then_max":
        per_layer_probs: List[mx.array] = []
        for slot, keys in zip(captured, layer_keys):
            if not slot or keys is None:
                continue
            q_layer = mx.concatenate(slot, axis=2)  # [1, n_heads, N, head_dim]
            head_dim = q_layer.shape[-1]
            per_layer_probs.append(_layer_softmax_probs(q_layer, keys, head_dim))
        importance = _aggregate_pool_then_max(per_layer_probs, pool_kernel)
    elif aggregation == "max_then_pool":
        per_layer: List[mx.array] = []
        for slot, keys in zip(captured, layer_keys):
            if not slot or keys is None:
                continue
            q_layer = mx.concatenate(slot, axis=2)
            head_dim = q_layer.shape[-1]
            per_layer.append(_compute_per_layer_importance(q_layer, keys, head_dim))
        stacked = _stack_layer_importances(per_layer)
        importance = _aggregate_importance(stacked)
        importance = _smooth_avg_pool(importance, pool_kernel)
    else:
        raise ValueError(
            f"Unknown aggregation {aggregation!r}; "
            "expected 'pool_then_max' or 'max_then_pool'."
        )

    mx.eval(importance)
    return _select_top_blocks(
        importance,
        M,
        block_size,
        keep_fraction,
        sink_size=sink_size,
        tail_keep=tail_keep,
    )


# ---------------------------------------------------------------------------
# Main-model cache population — "compact_with_tail"
# ---------------------------------------------------------------------------


def _apply_compact_with_tail(
    model: nn.Module,
    prompt: mx.array,
    prompt_cache: List[Any],
    keep_indices: mx.array,
    tail_size: int,
    prefill_step_size: int,
) -> mx.array:
    """Build the contiguous (history-tokens + tail-tokens) sequence and
    prefill it through the main model with normal positions.

    The selected "history" indices are those strictly less than
    ``M - tail_size``; the tail is taken verbatim from the prompt's last
    ``min(tail_size, M)`` tokens.

    Returns the ``prompt`` of length 1 that the caller should hand to the
    existing ``_step`` seed call (i.e. the last prompt token).
    """
    M = int(prompt.size)
    tail = min(tail_size, M)
    history_cut = M - tail

    if history_cut <= 0:
        merged = prompt
    else:
        idx = keep_indices.tolist()
        history = [i for i in idx if i < history_cut]
        if history:
            hist_tokens = prompt[mx.array(history, dtype=mx.int32)]
            tail_tokens = prompt[history_cut:]
            merged = mx.concatenate([hist_tokens, tail_tokens])
        else:
            merged = prompt[history_cut:]

    # Prefill everything except the last token; let _step handle the last.
    if merged.size > 1:
        _chunked_prefill(model, merged[:-1], prompt_cache, prefill_step_size)
    return merged[-1:]


# ---------------------------------------------------------------------------
# Main-model cache population — "original" (paper-faithful)
# ---------------------------------------------------------------------------


class _OriginalLayoutRopeWrapper:
    """Wraps an attention layer's RoPE for the lifetime of one
    SpecPrefill-enabled ``generate_step`` call when ``position_layout =
    "original"``.

    Two modes:

      * ``prefill_mode`` – the wrapper is called with an input of length
        ``L >= 1`` whose tokens correspond to ``keep_positions[offset :
        offset + L]`` (where ``offset`` is the cache offset at the start
        of the chunk). The wrapper rotates the input at those original
        positions via :func:`_rope_at_positions`.

      * decode mode (after ``finalize_prefill``) – the wrapper is called
        with ``L == 1`` and ``offset == cache_offset``; it shifts the
        offset by ``M - K`` (where ``K = |keep_positions|``) so the new
        query/key align with the contiguous decode positions starting at
        ``M``.
    """

    def __init__(self, inner, keep_positions: mx.array, M: int):
        self._inner = inner
        self._positions = keep_positions
        self._K = int(keep_positions.size)
        self._M = int(M)
        self._delta = self._M - self._K
        self.prefill_mode = True

    def finalize_prefill(self):
        self.prefill_mode = False

    def __call__(self, x, offset=0):
        if self.prefill_mode:
            L = x.shape[-2]
            positions = self._positions[offset : offset + L]
            return _rope_at_positions(x, positions, self._inner)
        # Decode: contiguous, with the spec-mandated offset shift.
        if isinstance(offset, int):
            adjusted = offset + self._delta
        else:
            adjusted = offset + self._delta
        return self._inner(x, offset=adjusted)


def _install_original_layout(
    model: nn.Module, keep_positions: mx.array, M: int
):
    handles: List[
        Tuple[Any, Any, _OriginalLayoutRopeWrapper]
    ] = []
    for layer in model.layers:
        info = _layer_attention_rope(layer)
        if info is None:
            continue
        attn, rope = info
        wrap = _OriginalLayoutRopeWrapper(rope, keep_positions, M)
        attn.rope = wrap
        handles.append((attn, rope, wrap))
    return handles


def _remove_original_layout(handles):
    for attn, orig_rope, _ in handles:
        attn.rope = orig_rope


def _apply_original_layout(
    model: nn.Module,
    prompt: mx.array,
    prompt_cache: List[Any],
    keep_indices: mx.array,
    prefill_step_size: int,
):
    """Install per-position RoPE wrappers, prefill the selected tokens
    (excluding the last one – left as the seed for ``_step``), then switch
    the wrappers to decode mode.

    Returns ``(seed_tokens, cleanup_fn)``: ``seed_tokens`` is ``prompt[-1:]``
    (the last prompt token, to be processed by the existing ``_step``
    call); ``cleanup_fn`` restores the original RoPE modules.
    """
    M = int(prompt.size)

    # The last prompt token (M-1) is always selected (we enforce it in
    # _select_top_blocks). It will be processed by _step, not the prefill.
    indices_list = keep_indices.tolist()
    if not indices_list or indices_list[-1] != M - 1:
        # Defensive: keep the last token.
        indices_list = sorted(set(indices_list + [M - 1]))
        keep_indices = mx.array(indices_list, dtype=mx.int32)

    handles = _install_original_layout(model, keep_indices, M)

    try:
        if len(indices_list) > 1:
            prefill_idx = mx.array(indices_list[:-1], dtype=mx.int32)
            prefill_tokens = prompt[prefill_idx]
            _chunked_prefill(
                model, prefill_tokens, prompt_cache, prefill_step_size
            )
    except Exception:
        _remove_original_layout(handles)
        raise

    for _, _, wrap in handles:
        wrap.finalize_prefill()

    def cleanup():
        _remove_original_layout(handles)

    return prompt[-1:], cleanup


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------


def apply_spec_prefill(
    model: nn.Module,
    speculator: nn.Module,
    prompt: mx.array,
    prompt_cache: List[Any],
    config: SpecPrefillConfig,
) -> Tuple[mx.array, Callable[[], None]]:
    """Run speculator-side scoring + populate ``prompt_cache`` for the main
    model.

    Args:
      model: the main model whose cache will be populated.
      speculator: the small scoring model.
      prompt: the original 1-D prompt token array (length ``M``).
      prompt_cache: the (empty) cache list for ``model``.
      config: tuning knobs.

    Returns:
      ``(seed, cleanup)``: ``seed`` is the 1-token ``mx.array`` that the
      caller should feed through ``_step`` to sample the first generated
      token; ``cleanup`` restores any model-level mutations
      (RoPE wrappers under the ``"original"`` layout).
    """
    keep_indices = compute_keep_indices(
        speculator,
        prompt,
        lookahead_steps=config.lookahead_steps,
        pool_kernel=config.pool_kernel,
        block_size=config.block_size,
        keep_fraction=config.keep_fraction,
        prefill_step_size=config.prefill_step_size,
        sink_size=config.sink_size,
        tail_keep=config.tail_size,
        aggregation=config.aggregation,
    )

    if config.position_layout == "compact_with_tail":
        seed = _apply_compact_with_tail(
            model,
            prompt,
            prompt_cache,
            keep_indices,
            config.tail_size,
            config.prefill_step_size,
        )

        def noop():
            return None

        return seed, noop

    if config.position_layout == "original":
        return _apply_original_layout(
            model,
            prompt,
            prompt_cache,
            keep_indices,
            config.prefill_step_size,
        )

    raise ValueError(
        f"Unknown position_layout {config.position_layout!r}; "
        "expected 'original' or 'compact_with_tail'."
    )


__all__ = [
    "SpecPrefillConfig",
    "apply_spec_prefill",
    "compute_keep_indices",
]
