# 26 - capstone

**Spec and rubric:** [chapter 26](../../chapters/26-capstone.md)

There is no harness here and no gate. You set the bar and you defend it.

## What goes in this folder

```
26-capstone/
  README.md          <- this file, leave it
  WRITEUP.md         <- copy WRITEUP_TEMPLATE.md and fill it in
  kernel.cu          <- or kernel.py for Triton
  ext.py / setup.py  <- the integration
  test_*.py          <- the test suite
  bench.py           <- your benchmark, to the chapter 24 methodology
  profile/           <- ncu reports, nsys traces, the evidence
```

## Before you write any code

Fill in the first three sections of `WRITEUP.md`: what is slow, the roofline
that proves it, and the speedup you predict. Date it. You will compare against
it at the end, and the gap between your prediction and your result is the most
instructive thing this course can give you at this point.

## The one rule

**No number goes in `WRITEUP.md` that you cannot reproduce from the commands in
`WRITEUP.md`.**

## Self-assessment

Score yourself against the rubric in chapter 26 before you consider it
finished. 70 is a pass. Note that analysis and benchmark honesty are 45 of the
100 points, which is deliberate: a well-understood 1.3x is worth more than an
unexplained 3x, because only the first one tells you how to get the next one.
