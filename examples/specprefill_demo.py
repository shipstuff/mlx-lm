# Copyright © 2026 mlx-lm contributors.
"""End-to-end demo of SpecPrefill.

Loads a main model and a smaller speculator, builds a long synthetic
prompt, and runs ``stream_generate`` with SpecPrefill enabled. Prints
prompt-tokens-per-second (TTFT proxy) and generation tps for both the
dense baseline and the SpecPrefill run.

Example::

    python examples/specprefill_demo.py \\
        --model /Users/Shared/models/exo/mlx-community--Qwen3.6-35B-A3B-4bit \\
        --speculator /Users/Shared/models/qwen3.5-0.8b-mlx-4bit \\
        --prompt-tokens 65536 \\
        --keep-fraction 0.3 \\
        --position-layout compact_with_tail
"""

import argparse
import time

import mlx.core as mx

from mlx_lm.generate import stream_generate
from mlx_lm.utils import load


def _synthetic_prompt(tokenizer, n_tokens: int) -> mx.array:
    """Build a prompt of ~``n_tokens`` tokens by repeating a paragraph of
    ordinary English and then planting a memorable phrase at the start
    plus a final question that asks to recall it.
    """
    secret = "The hidden code word is xylophone-rutabaga-velvet-43."
    filler = (
        "The quick brown fox jumps over the lazy dog. Mathematics is the "
        "language with which God has written the universe. To be or not "
        "to be, that is the question. All happy families are alike; "
        "each unhappy family is unhappy in its own way. The only thing "
        "we have to fear is fear itself. "
    )
    question = "What was the hidden code word at the start of this text?"

    text = secret + "\n\n"
    while len(tokenizer.encode(text)) < n_tokens - 64:
        text += filler
    text += "\n\n" + question
    ids = tokenizer.encode(text)
    return mx.array(ids[:n_tokens])


def _run(model, tokenizer, prompt, max_tokens, **stream_kwargs):
    text = []
    tic = time.perf_counter()
    last = None
    for resp in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, **stream_kwargs
    ):
        text.append(resp.text)
        last = resp
    toc = time.perf_counter()
    return "".join(text), last, toc - tic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--speculator", required=True)
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--keep-fraction",
        type=float,
        default=0.2,
        help=(
            "Top fraction of blocks to keep. Tested-robust low-density "
            "default; see SpecPrefillConfig.keep_fraction docstring for "
            "the knife-edge note."
        ),
    )
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--pool-kernel", type=int, default=13)
    p.add_argument("--lookahead-steps", type=int, default=8)
    p.add_argument(
        "--position-layout",
        choices=["original", "compact_with_tail"],
        default="compact_with_tail",
        help=(
            "compact_with_tail (default) keeps decode at dense baseline by "
            "compacting the kept history to contiguous RoPE positions and "
            "preserving the last --tail-size prompt tokens verbatim. "
            "original uses the paper-faithful non-contiguous position IDs "
            "and is materially slower to decode on Qwen3.6-class models."
        ),
    )
    p.add_argument("--tail-size", type=int, default=256)
    p.add_argument(
        "--sink-size",
        type=int,
        default=16,
        help="Always-keep leading prompt tokens (attention sink).",
    )
    p.add_argument(
        "--aggregation",
        choices=["pool_then_max", "max_then_pool"],
        default="pool_then_max",
        help=(
            "pool_then_max (default) smooths each (l, h, n) softmax slice "
            "along M before max-collapsing (L, H) and meaning over N. "
            "max_then_pool is the paper-faithful order – max over (L, H), "
            "mean over N, then smooth the resulting [M] vector. "
            "pool_then_max is empirically more robust at long contexts; "
            "max_then_pool is sharper and matches the paper exactly."
        ),
    )
    p.add_argument(
        "--baseline", action="store_true", help="Also run a dense baseline for comparison."
    )
    args = p.parse_args()

    print(f"Loading main model: {args.model}")
    model, tokenizer = load(args.model)

    print(f"Loading speculator: {args.speculator}")
    speculator, _ = load(args.speculator)

    print(f"Building synthetic prompt (~{args.prompt_tokens} tokens)…")
    prompt = _synthetic_prompt(tokenizer, args.prompt_tokens)
    print(f"  actual prompt length: {prompt.size}")

    if args.baseline:
        print("\n=== Dense baseline ===")
        text, last, wall = _run(
            model, tokenizer, prompt, args.max_tokens
        )
        if last is not None:
            print(
                f"  prefill: {last.prompt_tokens} tokens, "
                f"{last.prompt_tps:.1f} tps   "
                f"generation: {last.generation_tokens} tokens, "
                f"{last.generation_tps:.1f} tps"
            )
        print(f"  wall: {wall:.2f}s")
        print(f"  output[:200]: {text[:200]!r}")

    print(
        f"\n=== SpecPrefill (layout={args.position_layout}, "
        f"keep={args.keep_fraction}) ==="
    )
    text, last, wall = _run(
        model,
        tokenizer,
        prompt,
        args.max_tokens,
        speculator=speculator,
        speculator_threshold=1024,
        keep_fraction=args.keep_fraction,
        block_size=args.block_size,
        pool_kernel=args.pool_kernel,
        lookahead_steps=args.lookahead_steps,
        position_layout=args.position_layout,
        tail_size=args.tail_size,
        sink_size=args.sink_size,
        aggregation=args.aggregation,
    )
    if last is not None:
        print(
            f"  prefill: {last.prompt_tokens} tokens, "
            f"{last.prompt_tps:.1f} tps   "
            f"generation: {last.generation_tokens} tokens, "
            f"{last.generation_tps:.1f} tps"
        )
    print(f"  wall: {wall:.2f}s")
    print(f"  output[:200]: {text[:200]!r}")


if __name__ == "__main__":
    main()
