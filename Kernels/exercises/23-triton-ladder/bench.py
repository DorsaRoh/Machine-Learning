"""Harness for 23-triton-ladder. Do not edit.

Correctness against PyTorch, then speed against two things:
  - PyTorch's own kernel, which is a real baseline
  - your CUDA version from Parts I and II, if you fill in cuda_times.json

The gate needs cuda_times.json. Running the earlier exercises and copying four
numbers across is part of the exercise: the point is to find out what the
compiler cost you, and you cannot find that out against PyTorch alone.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

import kernels


def timeit(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def gbs(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def check(name, got, want, tol):
    if got is None:
        print(f"  {name:<12} NOT IMPLEMENTED")
        return False
    err = (got.double() - want.double()).abs()
    rel = err / want.double().abs().clamp_min(1e-6)
    ok = bool((err.max() <= tol) or (rel.max() <= tol))
    print(f"  {name:<12} max_abs {err.max():.3g}  max_rel {rel.max():.3g}  "
          f"{'ok' if ok else 'WRONG'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    dev = "cuda"
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(0))

    cuda_times = {}
    if os.path.exists("cuda_times.json"):
        cuda_times = json.load(open("cuda_times.json"))
    else:
        print("\ncuda_times.json not found -- speeds will be reported but NOT gated.")
        print("Copy it from cuda_times.example.json and fill in your own numbers.\n")

    n = 1 << 24
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    rows, cols = 8192, 1024
    m = torch.randn(rows, cols, device=dev)
    wide = torch.randn(64, 131072, device=dev)  # too wide for a one-shot row load
    w = torch.randn(cols, device=dev)
    b = torch.randn(cols, device=dev)

    print("\ncorrectness")
    results = {}
    ok = True

    try:
        got = kernels.add(x, y)
    except NotImplementedError:
        got = None
    ok &= check("add", got, x + y, 1e-5)

    try:
        got = kernels.total_sum(x)
    except NotImplementedError:
        got = None
    ok &= check("sum", got.reshape(()) if got is not None else None,
                x.double().sum().float(), 1e-3)

    try:
        got = kernels.softmax(m)
    except NotImplementedError:
        got = None
    ok &= check("softmax", got, torch.softmax(m, dim=-1), 1e-5)

    try:
        got = kernels.softmax(wide)
        ok &= check("softmax wide", got, torch.softmax(wide, dim=-1), 1e-5)
    except NotImplementedError:
        print("  softmax wide NOT IMPLEMENTED")
        ok = False

    try:
        got = kernels.layernorm(m, w, b)
    except NotImplementedError:
        got = None
    ok &= check("layernorm", got, F.layer_norm(m, (cols,), w, b, 1e-5), 1e-4)

    if not ok:
        print("\ncorrectness: FAIL -- fix this before looking at any timing.")
        return 1
    print("\ncorrectness: PASS")

    print("\nspeed (median ms)")
    print(f"{'kernel':<12} {'triton':>9} {'torch':>9} {'your cuda':>10} "
          f"{'GB/s':>8} {'vs cuda':>8}")
    cases = [
        ("add", lambda: kernels.add(x, y), lambda: x + y, 3 * n * 4),
        ("sum", lambda: kernels.total_sum(x), lambda: x.sum(), n * 4),
        ("softmax", lambda: kernels.softmax(m), lambda: torch.softmax(m, -1),
         2 * rows * cols * 4),
        ("layernorm", lambda: kernels.layernorm(m, w, b),
         lambda: F.layer_norm(m, (cols,), w, b, 1e-5), 2 * rows * cols * 4),
    ]
    gated = bool(cuda_times)
    passed = True
    for name, tri, ref, nbytes in cases:
        t = timeit(tri)
        r = timeit(ref)
        c = cuda_times.get(name)
        ratio = (c / t) if c else float("nan")
        results[name] = t
        print(f"{name:<12} {t:9.4f} {r:9.4f} "
              f"{(f'{c:10.4f}' if c else ' ' * 10)} {gbs(nbytes, t):8.1f} "
              f"{(f'{ratio:7.2f}x' if c else ' ' * 8)}")
        if c and ratio < 0.80:
            passed = False

    print("\n---------------------------------------------")
    if not gated:
        print("23-triton-ladder: correctness PASSED, speed NOT GATED")
        print("Fill in cuda_times.json to finish this exercise.")
        return 1
    print(f"23-triton-ladder: {'PASSED' if passed else 'NOT YET'} "
          f"(need >= 80% of your CUDA version on all four)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
