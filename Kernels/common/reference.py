#!/usr/bin/env python3
"""Golden-output generator for the Kernels masterclass.

Some exercises (softmax, LayerNorm, attention, GEMM in reduced precision) have
references that are fiddly to write correctly in C++ and easy to get subtly
wrong -- which would have you debugging the test instead of the kernel. For
those, PyTorch generates the inputs and the expected outputs once, as raw
little-endian float32 binaries, and the CUDA harness just reads them.

    python reference.py softmax --rows 4096 --cols 1024 --out data/
    python reference.py attention --batch 8 --heads 8 --seq 1024 --dim 64

Every op writes:
    data/<op>_<name>.bin   raw float32, C-contiguous
    data/<op>.json         manifest: shapes, dtypes, tolerances, flops, bytes

Determinism matters: the seed is fixed and the reference always accumulates in
float64 on CPU where precision is at stake, so the golden values are the
mathematically correct ones, not "what one particular GPU happened to do".
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:
    import torch
except ImportError:  # the references below only need torch for attention
    torch = None


SEED = 1234


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _save(out_dir: str, op: str, arrays: dict, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"op": op, "arrays": {}, **meta}
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        path = os.path.join(out_dir, f"{op}_{name}.bin")
        arr.tofile(path)
        manifest["arrays"][name] = {
            "file": os.path.basename(path),
            "shape": list(arr.shape),
            "dtype": "float32",
            "count": int(arr.size),
        }
    with open(os.path.join(out_dir, f"{op}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(arrays)} arrays for '{op}' to {out_dir}/")
    for name, info in manifest["arrays"].items():
        print(f"  {op}_{name}.bin  shape={info['shape']}")


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def op_softmax(args):
    rng = _rng()
    x = rng.standard_normal((args.rows, args.cols), dtype=np.float32) * args.scale
    # float64 accumulation + max subtraction: the numerically correct answer.
    xd = x.astype(np.float64)
    m = xd.max(axis=1, keepdims=True)
    e = np.exp(xd - m)
    y = e / e.sum(axis=1, keepdims=True)
    _save(
        args.out,
        "softmax",
        {"x": x, "y": y},
        {
            "rows": args.rows,
            "cols": args.cols,
            "tol": 1e-5,
            "bytes": 2 * x.nbytes,
            "flops": 0,
        },
    )


def op_layernorm(args):
    rng = _rng()
    n, d = args.rows, args.cols
    x = rng.standard_normal((n, d), dtype=np.float32)
    gamma = rng.standard_normal(d, dtype=np.float32)
    beta = rng.standard_normal(d, dtype=np.float32)
    xd = x.astype(np.float64)
    mu = xd.mean(axis=1, keepdims=True)
    var = xd.var(axis=1, keepdims=True)
    xhat = (xd - mu) / np.sqrt(var + args.eps)
    y = xhat * gamma + beta
    _save(
        args.out,
        "layernorm",
        {"x": x, "gamma": gamma, "beta": beta, "y": y,
         "mean": mu.squeeze(1), "rstd": (1.0 / np.sqrt(var + args.eps)).squeeze(1)},
        {"rows": n, "cols": d, "eps": args.eps, "tol": 1e-4,
         "bytes": 2 * x.nbytes, "flops": 0},
    )


def op_attention(args):
    if torch is None:
        raise SystemExit("attention reference needs torch: pip install torch")
    torch.manual_seed(SEED)
    b, h, s, d = args.batch, args.heads, args.seq, args.dim
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / (d ** 0.5)
    # float64 reference so your fp32 kernel is compared against the real answer.
    att = (q.double() @ k.double().transpose(-2, -1)) * scale
    if args.causal:
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
    p = att.softmax(dim=-1)
    o = p @ v.double()
    flops = 4.0 * b * h * s * s * d * (0.5 if args.causal else 1.0)
    _save(
        args.out,
        "attention",
        {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "o": o.float().numpy()},
        {"batch": b, "heads": h, "seq": s, "dim": d, "causal": bool(args.causal),
         "scale": scale, "tol": 2e-3, "flops": flops,
         "bytes": int(4 * 4 * b * h * s * d)},
    )


def op_gemm(args):
    rng = _rng()
    m, n, k = args.m, args.n, args.k
    a = (rng.standard_normal((m, k), dtype=np.float32) * 0.1)
    b = (rng.standard_normal((k, n), dtype=np.float32) * 0.1)
    c = (a.astype(np.float64) @ b.astype(np.float64))
    _save(
        args.out,
        "gemm",
        {"a": a, "b": b, "c": c},
        {"m": m, "n": n, "k": k, "tol": 1e-3,
         "flops": 2.0 * m * n * k,
         "bytes": int(a.nbytes + b.nbytes + 4 * m * n)},
    )


OPS = {
    "softmax": op_softmax,
    "layernorm": op_layernorm,
    "attention": op_attention,
    "gemm": op_gemm,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)

    p = sub.add_parser("softmax")
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--scale", type=float, default=1.0)

    p = sub.add_parser("layernorm")
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--eps", type=float, default=1e-5)

    p = sub.add_parser("attention")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--causal", action="store_true")

    p = sub.add_parser("gemm")
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--k", type=int, default=2048)

    for name in OPS:
        sub.choices[name].add_argument("--out", default="data")

    args = ap.parse_args()
    OPS[args.op](args)


if __name__ == "__main__":
    main()
