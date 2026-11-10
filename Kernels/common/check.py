#!/usr/bin/env python3
"""Compare a kernel's output against the golden reference, and say only that.

    python check.py data/softmax.json y out.bin --tol 1e-5

Prints a verdict plus an error profile: worst absolute error, worst relative
error, where it happened, and whether the errors are spread everywhere (a
formula bug) or concentrated (an indexing or boundary bug). It does not tell
you what the bug is. The shape of the error is the clue.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(path: str, shape) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    want = int(np.prod(shape))
    if arr.size != want:
        raise SystemExit(
            f"{path}: got {arr.size} float32 values, reference has {want}. "
            "Did your kernel write the whole output?"
        )
    return arr.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="e.g. data/softmax.json")
    ap.add_argument("array", help="name of the reference array, e.g. 'y'")
    ap.add_argument("actual", help="raw float32 .bin written by your kernel")
    ap.add_argument("--tol", type=float, default=None)
    args = ap.parse_args()

    with open(args.manifest) as f:
        man = json.load(f)
    info = man["arrays"][args.array]
    base = os.path.dirname(args.manifest)
    want = load(os.path.join(base, info["file"]), info["shape"])
    got = load(args.actual, info["shape"])
    tol = args.tol if args.tol is not None else man.get("tol", 1e-4)

    bad_finite = ~np.isfinite(got)
    abs_err = np.abs(got.astype(np.float64) - want.astype(np.float64))
    rel_err = abs_err / np.maximum(1e-6, np.abs(want.astype(np.float64)))
    fail_mask = (abs_err > tol) & (rel_err > tol)
    fail_mask |= bad_finite
    n_fail = int(fail_mask.sum())

    print(f"op            : {man['op']}  array '{args.array}' shape {info['shape']}")
    print(f"tolerance     : {tol:g}")
    print(f"max abs error : {np.nanmax(abs_err):.3g}")
    print(f"max rel error : {np.nanmax(rel_err):.3g}")
    if bad_finite.any():
        print(f"NaN/Inf       : {int(bad_finite.sum())} values")

    if n_fail == 0:
        print("verdict       : PASS")
        return 0

    frac = n_fail / want.size
    idx = np.argwhere(fail_mask)
    first = tuple(idx[0])
    print(f"verdict       : FAIL  ({n_fail} / {want.size} elements = {frac:.2%})")
    print(f"first bad at  : {first}  got {got[first]:.6g}, want {want[first]:.6g}")

    # Where the wrong values live is the most informative thing we can report
    # without handing over the answer.
    if want.ndim == 2:
        rows_bad = np.unique(idx[:, 0])
        cols_bad = np.unique(idx[:, 1])
        print(f"bad rows      : {len(rows_bad)} / {want.shape[0]}"
              f"   bad cols: {len(cols_bad)} / {want.shape[1]}")
    if frac > 0.9:
        print("pattern       : almost everything is wrong")
    elif frac < 0.02:
        print("pattern       : a small, localised set of elements is wrong")
    else:
        print("pattern       : a structured subset is wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
