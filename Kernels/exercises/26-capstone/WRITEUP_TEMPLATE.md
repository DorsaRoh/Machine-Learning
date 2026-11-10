# <project name>

<one sentence: what this kernel does and what it replaces>

---

## 1. What is slow, and why

Model / workload:
Profile evidence (op table or nsys screenshot, with the command that produced it):

The op(s) I am replacing account for ___% of step time.

### Roofline, before

```
bytes moved        :
FLOPs              :
arithmetic intensity:
ridge point of this GPU:
side of the ridge  :
theoretical minimum time:
measured time today:
headroom           :
```

### Prediction, written on <date> before any code

```
predicted kernel-level speedup :
predicted end-to-end speedup   :
what I expect the limiter to be after my change:
```

---

## 2. What I built

<the one or two ideas, in a paragraph. Not a code walkthrough.>

Why CUDA / why Triton:

What I deliberately did not support:

---

## 3. Results

### Environment

| | |
|---|---|
| GPU | |
| Driver / CUDA | |
| PyTorch | |
| Clocks locked? | |

### Methodology

- Warmup iterations:
- Measured iterations:
- Timing: CUDA events, synchronized
- Cache state between iterations:
- Baseline, and why it is the right baseline:

### Kernel level

| shape | dtype | baseline (ms) | mine (ms) | speedup | max rel error |
|---|---|---|---|---|---|
| | | | | | |

Median; p05 and p95 in `bench.py` output.

### End to end

| workload | baseline | mine | speedup |
|---|---|---|---|
| | | | |

### Where it does not help

<shapes, dtypes, batch sizes or hardware where this is neutral or worse, and
why. Every kernel has them. A write-up without this section is not
believable.>

### What I did not test

---

## 4. Analysis

### Optimization history

| version | change | predicted | measured | was I right? |
|---|---|---|---|---|
| v1 | naive | | | |
| v2 | | | | |

The rows where I was wrong, and what I had misunderstood:

### Final kernel, profiled

```
Speed of Light: __% memory, __% compute
dominant stall reason:
achieved bandwidth / TFLOP-s:
as a fraction of ceiling:
registers per thread:        spills:
occupancy: achieved __ / theoretical __
```

### The remaining gap

I am at ___% of the ceiling. The missing ___% is:

Evidence for that claim:

---

## 5. What did not work

<things tried and abandoned, with the measurement that killed each. This
section is worth more to the next person than section 2.>

---

## 6. Next

- What I would do with another week:
- What would need different hardware:
- What I now think I was wrong about at the start of this project:

---

## Reproducing

```bash
# exact commands, in order, that regenerate every number above
```
