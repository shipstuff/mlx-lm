"""Streaming MoE expert dispatch for models larger than available RAM.

When ``stream_experts`` is set in the model config, expert weights are not
held in GPU memory. Instead they are memory-mapped from the safetensors files
on disk and loaded on-demand during each forward pass. The OS page cache
manages which expert pages stay in DRAM vs. are paged from SSD.

This enables running MoE models whose expert weights exceed available unified
memory — for example Qwen3.5-397B-A17B (209 GB at 4-bit) on a 64 GB machine.

Usage::

    # Add to model's config.json:
    {"stream_experts": true}

    # Then load normally:
    model, tokenizer = mlx_lm.load("path/to/model")
"""

import json as _json
import re
import struct as _struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_expert_tensor(name: str) -> bool:
    """Return True if *name* belongs to a routed-expert weight tensor."""
    return "switch_mlp" in name or "mlp.experts." in name


def _mmap_tensor(file_path: str, tensor_name: str) -> Optional[np.ndarray]:
    """Memory-map a single tensor from a safetensors file."""
    with open(file_path, "rb") as fh:
        hdr_len = _struct.unpack("<Q", fh.read(8))[0]
        header = _json.loads(fh.read(hdr_len))
        data_start = 8 + hdr_len

    meta = header.get(tensor_name)
    if meta is None:
        return None

    offsets = meta["data_offsets"]
    shape = tuple(meta["shape"])
    dtype_str = meta["dtype"]

    _dtype_map = {
        "U32": np.uint32,
        "U8": np.uint8,
        "BF16": np.uint16,
        "F16": np.uint16,
        "F32": np.float32,
    }
    np_dtype = _dtype_map.get(dtype_str)
    if np_dtype is None:
        return None

    return np.memmap(
        file_path,
        dtype=np_dtype,
        mode="r",
        offset=data_start + offsets[0],
        shape=shape,
    )


# ---------------------------------------------------------------------------
# StreamingSwitchLinear
# ---------------------------------------------------------------------------


class StreamingSwitchLinear(nn.Module):
    """Drop-in replacement for :class:`QuantizedSwitchLinear` that loads
    expert weight slices from numpy memory-mapped safetensors files.

    On each call only the *K* selected expert slices are copied into
    ``mx.array`` objects for ``mx.gather_qmm``.  Hot experts remain in
    the OS page cache; cold ones are paged from NVMe.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._num_experts = num_experts
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self._np_weight: Optional[np.ndarray] = None
        self._np_scales: Optional[np.ndarray] = None
        self._np_biases: Optional[np.ndarray] = None

        # Placeholders so mlx's module bookkeeping is happy
        self.weight = mx.zeros((0,), dtype=mx.uint32)
        self.scales = mx.zeros((0,), dtype=mx.bfloat16)
        self.biases = mx.zeros((0,), dtype=mx.bfloat16)

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.freeze()

    # -- properties ----------------------------------------------------------

    @property
    def input_dims(self):
        return self._input_dims

    @property
    def output_dims(self):
        return self._output_dims

    @property
    def num_experts(self):
        return self._num_experts

    # -- weight management ---------------------------------------------------

    def set_mmap_weights(self, weight, scales, biases):
        """Attach numpy memory-mapped arrays for on-demand expert loading."""
        self._np_weight = weight
        self._np_scales = scales
        self._np_biases = biases

    def load_subset(self, unique_indices: List[int]):
        """Read the requested expert slices and return ``mx.array`` triples."""
        w = mx.array(self._np_weight[unique_indices])
        s_raw = mx.array(self._np_scales[unique_indices])
        # For affine mode: scales stored as bf16 (uint16) on disk, need float32.
        # For mxfp4 mode: scales stored as uint8, keep as-is.
        if self._np_scales.dtype == np.uint16:
            s = s_raw.view(mx.bfloat16).astype(mx.float32)
        else:
            s = s_raw

        b = None
        if self._np_biases is not None:
            b_raw = mx.array(self._np_biases[unique_indices])
            if self._np_biases.dtype == np.uint16:
                b = b_raw.view(mx.bfloat16).astype(mx.float32)
            else:
                b = b_raw
        return w, s, b

    # -- forward -------------------------------------------------------------

    @staticmethod
    def _gather_qmm(x, w, s, b, indices, group_size, bits, mode, sorted_indices):
        return mx.gather_qmm(
            x,
            w,
            s,
            b,
            rhs_indices=indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=sorted_indices,
        )

    def forward_with_subset(self, x, w, s, b, indices, sorted_indices=False):
        """Run ``gather_qmm`` with a pre-loaded expert subset."""
        out = self._gather_qmm(
            x,
            w,
            s,
            b,
            indices,
            self.group_size,
            self.bits,
            self.mode,
            sorted_indices,
        )
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out

    def __call__(self, x, indices, sorted_indices=False):
        mx.eval(indices)
        idx_flat = indices.reshape(-1).tolist()
        unique = sorted(set(idx_flat))
        w, s, b = self.load_subset(unique)
        remap = {orig: new for new, orig in enumerate(unique)}
        new_idx = mx.array([remap[i] for i in idx_flat], dtype=mx.uint32).reshape(
            indices.shape
        )
        return self.forward_with_subset(x, w, s, b, new_idx, sorted_indices)


# ---------------------------------------------------------------------------
# SwitchGLU patch — batch expert loading across gate/up/down projections
# ---------------------------------------------------------------------------


def _streaming_switchglu_call(self, x, indices):
    """Optimised ``SwitchGLU.__call__`` that evaluates expert indices once
    and loads all three projection subsets in a single batch."""
    x = mx.expand_dims(x, (-2, -3))

    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    if self.training:
        idx = mx.stop_gradient(idx)

    gate = self.gate_proj
    if isinstance(gate, StreamingSwitchLinear):
        # One eval + remap for all three projections
        mx.eval(idx)
        idx_flat = idx.reshape(-1).tolist()
        unique = sorted(set(idx_flat))
        remap = {orig: new for new, orig in enumerate(unique)}
        new_idx = mx.array([remap[i] for i in idx_flat], dtype=mx.uint32).reshape(
            idx.shape
        )

        gw, gs, gb = gate.load_subset(unique)
        uw, us, ub = self.up_proj.load_subset(unique)
        dw, ds, db = self.down_proj.load_subset(unique)

        x_gate = gate.forward_with_subset(x, gw, gs, gb, new_idx, do_sort)
        x_up = self.up_proj.forward_with_subset(x, uw, us, ub, new_idx, do_sort)
        x = self.down_proj.forward_with_subset(
            self.activation(x_up, x_gate),
            dw,
            ds,
            db,
            new_idx,
            do_sort,
        )
    else:
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)

    return x.squeeze(-2)


# ---------------------------------------------------------------------------
# Setup entry-point
# ---------------------------------------------------------------------------


def setup_streaming_experts(
    model: nn.Module,
    expert_weight_locations: Dict[str, str],
    model_path,
    config: dict,
):
    """Replace every ``QuantizedSwitchLinear`` in the model with a
    ``StreamingSwitchLinear`` backed by memory-mapped safetensors.

    The replacement is model-agnostic — it walks the module tree, finds
    ``QuantizedSwitchLinear`` instances inside ``SwitchGLU`` parents, and
    matches them to safetensors tensor names via the weight map index.

    Called automatically by :func:`load_model` when *stream_experts* is set.
    """
    quantization = config.get("quantization", config.get("quantization_config", {}))
    group_size = quantization.get("group_size", 64)
    bits = quantization.get("bits", 4)
    mode = quantization.get("mode", "affine")

    # Build tensor → file map from the safetensors index
    index_file = Path(model_path) / "model.safetensors.index.json"
    tensor_file_map: Dict[str, str] = {}
    if index_file.exists():
        with open(index_file) as fh:
            idx = _json.load(fh)
        for name, fname in idx.get("weight_map", {}).items():
            if _is_expert_tensor(name):
                tensor_file_map[name] = str(Path(model_path) / fname)

    # Build a reverse lookup: given a module path like
    # "language_model.model.layers[3].mlp.switch_mlp.gate_proj"
    # find the matching safetensors tensor name.
    # We key on (layer_number, projection_name) since those are
    # the only varying parts across model architectures.
    _tensor_by_layer_proj: Dict[Tuple[str, str], str] = {}
    for tname in tensor_file_map:
        # Extract layer number and projection name from tensor name
        # Works for any pattern like *.layers.N.*.switch_mlp.PROJ.weight
        m = re.search(r"layers\.(\d+)\b.*switch_mlp\.(\w+_proj)\.weight$", tname)
        if m:
            _tensor_by_layer_proj[(m.group(1), m.group(2))] = tname

    count = 0

    def _replace_in_switchglu(parent, path):
        nonlocal count
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            child = parent.get(proj_name)
            if not isinstance(child, QuantizedSwitchLinear):
                continue

            # Extract layer number from the module path
            layer_match = re.search(r"\[(\d+)\]", path)
            if not layer_match:
                continue
            layer_num = layer_match.group(1)

            # Look up the tensor name for this (layer, projection) pair
            w_key = _tensor_by_layer_proj.get((layer_num, proj_name))
            if w_key is None:
                continue

            w_file = tensor_file_map[w_key]
            s_key = w_key.replace(".weight", ".scales")
            b_key = w_key.replace(".weight", ".biases")

            w_mmap = _mmap_tensor(w_file, w_key)
            s_mmap = _mmap_tensor(tensor_file_map.get(s_key, w_file), s_key)
            b_mmap = _mmap_tensor(tensor_file_map.get(b_key, w_file), b_key)

            if w_mmap is None or s_mmap is None:
                continue

            num_experts = w_mmap.shape[0]
            output_dims = w_mmap.shape[1]
            input_dims = s_mmap.shape[2] * group_size

            streaming = StreamingSwitchLinear(
                input_dims=input_dims,
                output_dims=output_dims,
                num_experts=num_experts,
                bias=child.get("bias") is not None,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            streaming.set_mmap_weights(w_mmap, s_mmap, b_mmap)

            if child.get("bias") is not None:
                streaming.bias = child.bias

            parent[proj_name] = streaming
            count += 1

    def _walk(module, path=""):
        for name in list(module.keys()):
            child = module[name]
            if isinstance(child, SwitchGLU):
                _replace_in_switchglu(child, f"{path}.{name}")
            elif isinstance(child, nn.Module):
                _walk(child, f"{path}.{name}")
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        _walk(item, f"{path}.{name}[{i}]")

    _walk(model)

    if count > 0:
        SwitchGLU.__call__ = _streaming_switchglu_call

    print(f"[streaming] {count} expert layers → StreamingSwitchLinear (mmap)")
