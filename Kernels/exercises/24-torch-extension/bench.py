"""Harness for 24-torch-extension. Do not edit.

This harness is itself an example of the chapter 24 methodology: warmup,
events, interleaved baseline and candidate, median and spread, cache state
stated, and an end-to-end number rather than a kernel-only one.

Read it. You will write something like it for the capstone.
"""

import statistics
import sys

import torch

import ext


def timeit(fn, warmup=25, iters=100):
    """Events, synchronize, median. Returns (median_ms, p05, p95)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    ts.sort()
    return statistics.median(ts), ts[int(0.05 * len(ts))], ts[int(0.95 * len(ts)) - 1]


def chain(fn, x, w, b, layers):
    """The end-to-end workload: the memory-bound subgraph of a transformer
    block, repeated. No matmuls, deliberately -- this measures the thing your
    kernel can actually affect."""
    y = x
    for _ in range(layers):
        y = fn(y, w, b)
    return y


def main():
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    dev = "cuda"
    torch.manual_seed(0)
    rows, cols, layers = 8192, 1024, 24
    print(torch.cuda.get_device_name(0))
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")

    x = torch.randn(rows, cols, device=dev)
    w = torch.randn(cols, device=dev)
    b = torch.randn(cols, device=dev)

    # ---- correctness ----------------------------------------------------
    try:
        got = ext.fused(x, w, b)
    except NotImplementedError:
        print("\next.fused is not implemented yet.")
        return 1
    want = ext.eager(x, w, b)
    err = (got.double() - want.double()).abs().max().item()
    rel = ((got.double() - want.double()).abs() /
           want.double().abs().clamp_min(1e-6)).max().item()
    correct = err <= 1e-4 or rel <= 1e-4
    print(f"\ncorrectness    : {'PASS' if correct else 'FAIL'} "
          f"(max_abs {err:.3g}, max_rel {rel:.3g}, tol 1e-4)")

    # shifted distribution: catastrophic cancellation if you used sum/sumsq
    xs = torch.randn(rows, cols, device=dev) + 1000.0
    e2 = (ext.fused(xs, w, b).double() - ext.eager(xs, w, b).double()).abs().max().item()
    shifted_ok = e2 <= 1e-3
    print(f"shifted mean   : {'PASS' if shifted_ok else 'FAIL'} (max_abs {e2:.3g})")
    correct = correct and shifted_ok

    # ---- graph check ----------------------------------------------------
    try:
        compiled_op = torch.compile(lambda t: ext.fused(t, w, b), fullgraph=True)
        compiled_op(x)
        torch.cuda.synchronize()
        no_break = True
    except Exception as e:  # noqa: BLE001
        no_break = False
        print(f"fullgraph      : FAIL -- {type(e).__name__}: {str(e)[:160]}")
    if no_break:
        print("fullgraph      : PASS (torch.compile traced through your op)")

    if not correct:
        print("\nFix correctness before reading any timing number.")
        return 1

    # ---- speed ----------------------------------------------------------
    eager_fn = lambda: chain(ext.eager, x, w, b, layers)       # noqa: E731
    mine_fn = lambda: chain(ext.fused, x, w, b, layers)        # noqa: E731
    compiled = torch.compile(lambda t: chain(ext.eager, t, w, b, layers))
    comp_fn = lambda: compiled(x)                              # noqa: E731

    # interleave so all three see the same thermal state
    for _ in range(5):
        eager_fn(); mine_fn(); comp_fn()
    torch.cuda.synchronize()

    e_med, e_lo, e_hi = timeit(eager_fn)
    m_med, m_lo, m_hi = timeit(mine_fn)
    c_med, c_lo, c_hi = timeit(comp_fn)

    bytes_min = 2 * rows * cols * 4 * layers  # read x, write out, per layer
    print(f"\nworkload       : {layers} layers of [{rows}, {cols}] fp32")
    print("cache state    : tensors are {:.0f} MB, re-read each call (hot in L2 "
          "on large-L2 parts)".format(rows * cols * 4 / 1e6))
    print(f"{'':<15}{'median':>9}{'p05':>9}{'p95':>9}{'GB/s':>9}")
    for name, (med, lo, hi) in (("eager", (e_med, e_lo, e_hi)),
                                ("torch.compile", (c_med, c_lo, c_hi)),
                                ("yours", (m_med, m_lo, m_hi))):
        print(f"{name:<15}{med:9.3f}{lo:9.3f}{hi:9.3f}"
              f"{bytes_min / (med * 1e-3) / 1e9:9.1f}")
    print(f"\nminimum traffic for this workload: {bytes_min / 1e6:.0f} MB")
    print(f"speedup vs eager         : {e_med / m_med:.2f}x")
    print(f"speedup vs torch.compile : {c_med / m_med:.2f}x")
    if m_hi / m_med > 1.5:
        print("WARNING: your p95 is >1.5x your median. Something else is running,"
              " or your clocks are drifting. Do not trust these numbers yet.")

    passed = correct and no_break and (e_med / m_med >= 1.3)
    print("\n---------------------------------------------")
    print(f"24-torch-extension: {'PASSED' if passed else 'NOT YET'} "
          "(need correct, fullgraph-clean, >= 1.30x eager)")
    if passed and c_med / m_med < 1.0:
        print("\nNote: torch.compile beats you. You pass the gate, but chapter 24")
        print("says the baseline is the best thing a competent person would use.")
        print("Write down why it wins before you call this finished.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
