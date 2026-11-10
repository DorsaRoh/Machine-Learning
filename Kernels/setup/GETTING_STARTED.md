# Getting a GPU and running your first exercise

You cannot learn this on a Mac. Apple silicon has a fine GPU, but CUDA does not
run on it, and the whole point of this course is to reason about a specific
machine: NVIDIA's streaming multiprocessor. So the first skill is renting one
and getting a compile-run-profile loop that takes seconds, not minutes.

## Pick a GPU

| Option | Cost | GPU | Good for |
|---|---|---|---|
| [Google Colab](https://colab.research.google.com) (free) | free, preemptible | T4 (sm_75) | Parts 0-II, all of Part IV except Hopper |
| [Google Colab Pro](https://colab.research.google.com/signup) | ~$10/mo | L4 / A100 | everything up to Part V |
| [Lightning Studios](https://lightning.ai) | free tier hours | T4 / A10G | long sessions, persistent disk |
| [Modal](https://modal.com) | pay per second | T4 -> H100 | running one exercise for 40 seconds and paying $0.01 |
| [vast.ai](https://cloud.vast.ai) / [RunPod](https://www.runpod.io) | $0.2-$2/hr | anything incl. H100 | Part III Tensor Cores, Part V, Hopper features |

Only vast.ai, RunPod and Lambda give you root, which is what Nsight Compute
needs to read performance counters. From Chapter 16 onward that stops being
optional.

Start on a free T4. Everything through Chapter 12 is instructive on a T4. You
need Ampere (A100/A10/RTX 30xx, sm_80+) for `cp.async` and bf16, and Hopper
(H100, sm_90) for TMA and warp specialization. Those chapters say so.

Rent the big GPU only when a chapter demands it. A kernel engineer who can only
work on an H100 is not a kernel engineer.

**One exception needs two GPUs:** `25-allreduce`, and chapter 25's exercise
25.1 which asks you to measure your own interconnect. A 2x or 4x instance on
vast.ai or RunPod for an hour costs a couple of dollars, and having measured an
NVLink and a PCIe link once is worth more than reading about either. Check
`nvidia-smi topo -m` as soon as the instance boots -- some cheap multi-GPU
instances are PCIe-only, which is itself worth seeing.

**Two exercises need PyTorch rather than just `nvcc`:** `23-triton-ladder`
(`pip install triton`, bundled with recent PyTorch) and `24-torch-extension`.
Colab has both already.

## Colab, the 60-second path

1. Open a new notebook, then `Runtime -> Change runtime type -> T4 GPU`.
2. Paste this into the first cell:

```python
!nvidia-smi
!nvcc --version
!git clone https://github.com/DorsaRoh/Machine-Learning.git
%cd Machine-Learning/Kernels
```

3. Build and run an exercise:

```python
%cd /content/Machine-Learning/Kernels/exercises/02-vector-add
!make run
```

`setup/colab_bootstrap.ipynb` in this folder does the same thing with a device
report and a helper for running any exercise by name.

Colab already ships the CUDA toolkit, so `nvcc` just works. It does **not**
ship Nsight Compute by default; Chapter 16 shows you how to install it and what
to do when the environment will not let you profile (Colab blocks the
performance counters some of the time -- that is what Modal and vast.ai are
for).

## Modal, the path that scales to an H100

```bash
pip install modal && modal setup
```

`setup/modal_runner.py` (created in Chapter 16) mounts this folder into a
container with a GPU of your choosing and runs `make run` in it. The useful
property is that switching from a T4 to an H100 is a one-word edit, so
"does my kernel still win on newer hardware?" costs you a minute.

## Local checks you *can* do on the Mac

**The entire prerequisites part runs here, with no setup at all.** Nine
chapters and six self-checking exercises covering pointers, flat indexing,
floating point, the toolchain, the math and performance estimation — plain
C++, no CUDA, no GPU:

```bash
cd Kernels/prerequisites/exercises/p0-hello
make run
```

If that prints a list of `[FAIL]` lines, your toolchain is working correctly
and you have your first exercise. See [Part -1](../prerequisites/README.md),
which opens with a nine-question test for deciding whether you need it.

Beyond that, without a GPU you can still:

- Read and reason about SASS and PTX from a saved dump.
- Run the NumPy/PyTorch reference generators in `common/reference.py`.
- Do the pencil-and-paper exercises: every chapter has them, and they are the
  ones that actually build the cost model in your head.
- Prototype any kernel's *algorithm* in C++ first. Writing online softmax or a
  blocked matmul on a CPU, where you have a debugger and `printf` works, is
  far cheaper than meeting it for the first time inside a CUDA kernel. The
  `p4-math` and `p5-estimate` exercises are built around exactly this habit.

A surprising amount of real kernel work is arithmetic done before you type.

## Verify your setup

```bash
cd Kernels/exercises/00-hello-gpu
make run
```

You should see a device report: SM count, registers per SM, shared memory per
SM, L2 size, and a *measured* streaming bandwidth number. Write that measured
bandwidth down. Every memory-bound performance gate in this course is a
percentage of it.

## Toolchain glossary

| Command | What it does | When you reach for it |
|---|---|---|
| `nvcc` | compiles `.cu` to a fat binary (host code + PTX + SASS) | always |
| `nvidia-smi` | what GPU, what driver, who is using it | first thing, every session |
| `compute-sanitizer` | memcheck / racecheck / initcheck / synccheck | the moment a result is wrong *sometimes* |
| `ncu` (Nsight Compute) | per-kernel counters, speed-of-light, stall reasons | the moment a result is right but slow |
| `nsys` (Nsight Systems) | timeline across kernels, copies, CPU | when the gaps *between* kernels are the problem |
| `cuobjdump -sass` | the actual machine code | when the profiler's answer stops being enough |
| `ptxas -v` | register and shared-memory usage per kernel | checking occupancy without running anything |

## Build flags you should recognise

The shared `common/common.mk` compiles with:

```
-O3 -std=c++17 -arch=native -lineinfo
```

- `-arch=native` targets exactly the GPU present. If you compile for the wrong
  architecture you either get a JIT penalty at launch or a hard failure.
- `-lineinfo` maps SASS instructions back to your source lines so Nsight can
  attribute stalls per line. It does not slow the kernel down. Always use it.
- `-O3` affects host code. Device code is optimised by `ptxas` regardless, but
  `-G` (debug) **disables** device optimisation entirely -- never benchmark a
  `-G` build. It is 10-50x slower and will teach you false lessons.
