"""Correctness tests for 24-torch-extension. Do not edit.

Run with `python test_ext.py` or `pytest test_ext.py`.

These are the tests a reviewer would ask for: edge shapes, inputs PyTorch
actually hands you, and the numerically awkward cases. A kernel that passes
bench.py and fails these is not shipped, it is demoed.
"""

import sys

import torch

import ext

FAIL = []


def check(name, x, w, b, tol=1e-4):
    try:
        got = ext.fused(x, w, b)
    except Exception as e:  # noqa: BLE001
        FAIL.append(f"{name}: raised {type(e).__name__}: {e}")
        print(f"  {name:<28} RAISED {type(e).__name__}")
        return
    want = ext.eager(x, w, b)
    if got.shape != want.shape:
        FAIL.append(f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        print(f"  {name:<28} WRONG SHAPE {tuple(got.shape)}")
        return
    err = (got.double() - want.double()).abs().max().item()
    ok = err <= tol
    if not ok:
        FAIL.append(f"{name}: max_abs {err:.3g} > {tol:g}")
    print(f"  {name:<28} max_abs {err:.3g}  {'ok' if ok else 'WRONG'}")


def main():
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    torch.manual_seed(1)
    d = "cuda"

    print("shapes")
    for rows, cols in [(1, 1), (1, 1024), (8192, 1), (3, 17), (1000, 333),
                       (8192, 1024), (17, 4096), (4, 8192), (2, 32768)]:
        x = torch.randn(rows, cols, device=d)
        check(f"[{rows}, {cols}]", x, torch.randn(cols, device=d),
              torch.randn(cols, device=d))

    print("\ninputs PyTorch will actually hand you")
    base = torch.randn(256, 2048, device=d)
    w = torch.randn(1024, device=d)
    b = torch.randn(1024, device=d)
    check("non-contiguous slice", base[:, :1024], w, b)
    check("transposed view", torch.randn(1024, 256, device=d).t(), w, b)
    wide = torch.randn(4, 256, 1024, device=d)
    check("3D input", wide, w, b)

    print("\nnumerics")
    cols = 1024
    w = torch.randn(cols, device=d)
    b = torch.randn(cols, device=d)
    check("mean 0, var 1", torch.randn(512, cols, device=d), w, b)
    check("mean 1000", torch.randn(512, cols, device=d) + 1000.0, w, b, 1e-3)
    check("mean 1e6", torch.randn(512, cols, device=d) + 1e6, w, b, 1e-1)
    check("tiny variance", torch.randn(512, cols, device=d) * 1e-6, w, b)
    check("constant row", torch.ones(512, cols, device=d), w, b)

    print("\n---------------------------------------------")
    if FAIL:
        print(f"{len(FAIL)} failure(s):")
        for f in FAIL:
            print("  " + f)
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
