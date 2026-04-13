# Copyright © 2024 Apple Inc.
import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.streaming_switch import (
    StreamingSwitchLinear,
    _streaming_switchglu_call,
    setup_streaming_experts,
)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU


class TestStreamingSwitchLinear(unittest.TestCase):
    """Test that StreamingSwitchLinear produces the same output as
    QuantizedSwitchLinear when given identical weights."""

    def _make_quantized_layer(self, num_experts=8, out_dim=128, in_dim=256):
        """Create a QuantizedSwitchLinear with random quantized weights."""
        ql = QuantizedSwitchLinear(
            in_dim, out_dim, num_experts, bias=False, group_size=64, bits=4
        )
        # Quantize random weights
        scale = (1 / in_dim) ** 0.5
        w = mx.random.uniform(-scale, scale, (num_experts, out_dim, in_dim))
        ql.weight, ql.scales, ql.biases = mx.quantize(w, group_size=64, bits=4)
        ql.freeze()
        return ql

    def _make_streaming_from_quantized(self, ql):
        """Create a StreamingSwitchLinear from a QuantizedSwitchLinear's weights."""
        ssl = StreamingSwitchLinear(
            input_dims=ql.input_dims,
            output_dims=ql.output_dims,
            num_experts=ql.num_experts,
            bias="bias" in ql,
            group_size=ql.group_size,
            bits=ql.bits,
            mode=ql.mode,
        )
        # Use numpy views of the mlx arrays (simulating mmap)
        # For bf16 scales/biases: view mlx array as uint16 first, then to numpy
        ssl.set_mmap_weights(
            np.array(ql.weight),
            np.array(ql.scales.astype(mx.bfloat16).view(mx.uint16)),
            (
                np.array(ql.biases.astype(mx.bfloat16).view(mx.uint16))
                if ql.get("biases") is not None
                else None
            ),
        )
        return ssl

    def test_streaming_matches_quantized(self):
        """StreamingSwitchLinear output matches QuantizedSwitchLinear."""
        ql = self._make_quantized_layer()
        ssl = self._make_streaming_from_quantized(ql)

        x = mx.random.normal((1, 1, 1, 256))
        indices = mx.array([[2, 5]])

        out_ql = ql(x, indices)
        out_ssl = ssl(x, indices)

        mx.eval(out_ql, out_ssl)
        self.assertTrue(
            mx.allclose(out_ql, out_ssl, atol=1e-2).item(),
            f"Outputs differ: ql={out_ql}, ssl={out_ssl}",
        )

    def test_streaming_multiple_tokens(self):
        """Streaming works with multiple tokens."""
        ql = self._make_quantized_layer(num_experts=16, out_dim=128, in_dim=256)
        ssl = self._make_streaming_from_quantized(ql)

        x = mx.random.normal((1, 4, 1, 256))
        indices = mx.array([[0, 3, 7, 12]] * 4)

        out_ql = ql(x, indices)
        out_ssl = ssl(x, indices)

        mx.eval(out_ql, out_ssl)
        self.assertTrue(mx.allclose(out_ql, out_ssl, atol=1e-2).item())

    def test_switchglu_streaming_patch(self):
        """Patched SwitchGLU with streaming layers matches original."""
        # hidden_dim must be >= group_size for down_proj quantization
        in_dim, hidden_dim, num_experts = 256, 256, 8

        # Build a SwitchGLU with quantized layers
        glu = SwitchGLU(in_dim, hidden_dim, num_experts, bias=False)
        glu.gate_proj = glu.gate_proj.to_quantized(group_size=64, bits=4)
        glu.up_proj = glu.up_proj.to_quantized(group_size=64, bits=4)
        glu.down_proj = glu.down_proj.to_quantized(group_size=64, bits=4)

        # Get reference output
        x = mx.random.normal((1, 1, in_dim))
        indices = mx.array([[1, 4]])
        ref_out = glu(x, indices)
        mx.eval(ref_out)

        # Now replace with streaming layers using the helper
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            ql = glu[proj_name]
            ssl = self._make_streaming_from_quantized(ql)
            glu[proj_name] = ssl

        # Apply the streaming SwitchGLU patch
        SwitchGLU.__call__ = _streaming_switchglu_call

        stream_out = glu(x, indices)
        mx.eval(stream_out)

        self.assertTrue(
            mx.allclose(ref_out, stream_out, atol=1e-2).item(),
            f"SwitchGLU outputs differ after streaming patch",
        )


if __name__ == "__main__":
    unittest.main()
