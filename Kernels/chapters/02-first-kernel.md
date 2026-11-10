# Chapter 02 - Your first kernel

> The mechanics. Small chapter, high density, and every single thing in it is
> load-bearing for the rest of the course.

---

## 1. Two machines, two memories

A CUDA program runs on two computers that share nothing. The CPU (**host**) has
its own memory. The GPU (**device**) has its own memory. A pointer from one is
meaningless in the other, and dereferencing a host pointer on the device is a
segfault that presents as `an illegal memory access was encountered`.

The shape of every CUDA program:

```
allocate on device  ->  copy in  ->  launch kernel  ->  copy out  ->  free
```

```cuda
float *d_a;
cudaMalloc(&d_a, n * sizeof(float));                            // device alloc
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice); // in
my_kernel<<<grid, block>>>(d_a, n);                              // run
cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost); // out
cudaFree(d_a);
```

The `d_` / `h_` prefix convention is not decoration. Keep it. You will mix them
up, and the compiler will not stop you, because both are just `float*`.

### The transfer is expensive

PCIe gen4 x16 moves ~25 GB/s. Your GPU's own memory moves 2000 GB/s. **Host-
device transfer is roughly 80x slower than device-local memory.** This has one
enormous consequence for how you structure programs:

> Move data to the GPU once and keep it there. A kernel that is 10x faster than
> the CPU is a loss if you copy the data in and out each call.

This is why PyTorch has `.to('cuda')` as an explicit, visible operation, and
why a model that lives on the GPU for an entire training run is the norm.

---

## 2. Writing the kernel

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

Line by line, because every line is a lesson.

**`__global__`** - runs on the device, called from the host. Must return
`void`; results come out through pointers. The other qualifiers:

| Qualifier | Runs on | Called from |
|---|---|---|
| `__global__` | device | host (and device, dynamic parallelism) |
| `__device__` | device | device only |
| `__host__` | host | host |
| `__host__ __device__` | both | both (compiled twice) |

**`blockIdx.x * blockDim.x + threadIdx.x`** - the global index. Three built-in
variables, all of type `uint3`, all set by the hardware:

```
threadIdx  position within the block     [0, blockDim)
blockIdx   position within the grid      [0, gridDim)
blockDim   threads per block             you chose this
gridDim    blocks per grid               you chose this
```

A grid of 4 blocks x 8 threads, laid out:

```
block 0: [ t0 t1 t2 t3 t4 t5 t6 t7 ]   global 0..7
block 1: [ t0 t1 t2 t3 t4 t5 t6 t7 ]   global 8..15
block 2: [ t0 t1 t2 t3 t4 t5 t6 t7 ]   global 16..23
block 3: [ t0 t1 t2 t3 t4 t5 t6 t7 ]   global 24..31
```

**`if (i < n)`** - the bounds guard. You launch `ceil(n / block)` blocks, so
the last block has threads with `i >= n`. Without the guard they write past the
end of the array. CUDA does not bounds-check. The symptom is anything from
nothing at all (you wrote into padding) to silent corruption of an unrelated
tensor three kernels later. Run `compute-sanitizer` and it tells you
immediately.

**`const float*`** - not just hygiene. `const` plus `__restrict__` lets the
compiler prove there is no aliasing, which unlocks the read-only data path and
lets it keep values in registers across stores. It is occasionally worth a
double-digit percentage, for free.

---

## 3. Launching it

```cuda
int block = 256;
int grid  = (n + block - 1) / block;   // ceil-divide: memorise this idiom
add<<<grid, block>>>(d_a, d_b, d_c, n);
```

The `<<<...>>>` takes up to four arguments:

```cuda
kernel<<<grid, block, shared_bytes, stream>>>(args...);
```

- **grid** - blocks, as `int` or `dim3` for 2D/3D.
- **block** - threads per block, max 1024. Make it a multiple of 32 (the warp
  size) always; 256 is a sane default and you tune it in Chapter 04.
- **shared_bytes** - dynamic shared memory, Chapter 06. Default 0.
- **stream** - which queue, Chapter 17. Default the null stream.

### Starting values

- **block = 256.** A multiple of 32, enough warps for the scheduler to choose
  from, small enough that several blocks fit per SM.
- **grid = ceil(n / block).** One thread per element.

Both get revisited. Neither is ever far wrong.

### 2D and 3D

```cuda
dim3 block(16, 16);                                   // 256 threads, as 16x16
dim3 grid((width + 15) / 16, (height + 15) / 16);
kernel<<<grid, block>>>(...);

// inside:
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) out[y * width + x] = ...;
```

This is *purely notational convenience*. The hardware flattens it. But the
flattening order matters enormously, and it is the first thing people get
wrong:

> **x is the fastest-varying dimension.** Thread `(0,0)`, `(1,0)`, `(2,0)` ...
> are consecutive in the warp. Thread `(0,1)` is 16 lanes later.

So in a 16x16 block, one warp covers two full rows of x. If you index memory as
`out[x * height + y]`, consecutive lanes are `height` elements apart and your
kernel moves 32x the bytes it should. Chapter 03 is about exactly this.

### Grid-stride loops

Instead of one thread per element, a fixed grid where each thread loops:

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}
```

Why this is usually better:

- **Any n works** with any grid size; no launch-size arithmetic coupling.
- **You control the grid**, so you can size it to the GPU
  (`sm_count * blocks_per_sm`) and avoid tail effects.
- **Amortises setup**: per-thread index math and pointer setup happen once for
  many elements.
- **Debuggable**: launch with grid=1, block=1 and it becomes a serial loop that
  produces the right answer. Enormously useful.

The stride is `blockDim.x * gridDim.x` -- the total thread count -- so
consecutive threads still touch consecutive addresses on every iteration.
Coalescing is preserved. A loop striding by 1 per thread (`i = tid * k + j`)
destroys it.

---

## 4. Errors, and why they appear in the wrong place

**Kernel launches are asynchronous.** `<<<>>>` returns to the CPU immediately,
usually before the GPU has started. It also returns `void`. So:

```cuda
my_kernel<<<grid, block>>>(args);   // error here...
cudaMemcpy(...);                    // ...reported here
```

Two different errors are possible and you need both checks:

```cuda
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // launch-time: bad config, too much smem
CUDA_CHECK(cudaDeviceSynchronize());   // run-time: illegal access, assert
```

`cudaGetLastError()` catches "this launch was invalid". Only synchronising
catches "the kernel faulted while running". The `CUDA_CHECK_LAUNCH()` macro in
`bench.cuh` does both.

In production you remove the sync (it serialises everything); in development
you keep it, because a kernel error that surfaces three calls later will cost
you an hour. Common launch-time errors:

| Message | Means |
|---|---|
| `invalid configuration argument` | block > 1024, or grid dim > limit, or 0 |
| `invalid device function` | compiled for the wrong `-arch` |
| `too many resources requested for launch` | registers x threads > the SM's file |
| `an illegal memory access was encountered` | out of bounds, or a host pointer |
| `misaligned address` | a `float4` load from an address not 16-byte aligned |

---

## 5. printf from a kernel

It works, and it is the right first debugging tool:

```cuda
if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("block %d: i=%d val=%f\n", blockIdx.x, i, val);
```

Guard it. Without the guard you get a million interleaved lines and a
noticeably slower kernel. Output is buffered and appears at the next
synchronisation point. For index bugs, `printf` from one thread at `n = 64`
beats any amount of staring.

---

## 6. Unified memory, and why not to start there

```cuda
float* a;
cudaMallocManaged(&a, n * sizeof(float));
a[0] = 1.0f;              // works on the host
kernel<<<g,b>>>(a, n);    // works on the device
cudaDeviceSynchronize();
printf("%f", a[0]);       // works on the host again
```

The driver migrates pages on demand. It is genuinely convenient, and it hides
exactly the thing this course is about. A page fault on first touch costs
microseconds; migration thrash can make a kernel 10x slower with no visible
cause in your code. Use explicit `cudaMalloc`/`cudaMemcpy` while learning, so
that data movement stays something you decided rather than something that
happened to you.

---

## 7. A complete program

```cuda
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while(0)

__global__ void add(const float* __restrict__ a, const float* __restrict__ b,
                    float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int block = 256, grid = (n + block - 1) / block;
    add<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    printf("c[0] = %f (expect 3)\n", h_c[0]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

Note the final `cudaMemcpy` doubles as a synchronisation point: it blocks until
the kernel finishes. That is why this program works without an explicit sync,
and why people who later switch to async copies are suddenly confused.

---

## 8. Common failure modes

Symptoms, not solutions. Match the symptom, then go think.

**Output is all zeros, or all the poison value.** Nothing wrote there. Grid too
small? Kernel never launched (check `cudaGetLastError`)? Wrong `-arch`, so the
launch silently failed?

**The first `blockDim` elements are right, the rest are wrong.** Your index is
missing the block term.

**Everything is right except the last few elements.** Integer division truncated
your grid size.

**Correct at n=1024, wrong at n=1000.** The bounds guard, or its absence.

**Correct most runs, wrong sometimes.** A race, or an out-of-bounds write that
usually lands somewhere harmless. `compute-sanitizer`, now.

**`illegal memory access` at a `cudaMemcpy` nowhere near the kernel.** The
kernel faulted; the error merely surfaced here. Add a sync after the launch to
find the real location.

**Host array unchanged after the kernel.** You forgot the device-to-host copy,
or copied in the wrong direction, or passed a host pointer to the kernel.

**It is far slower than you predicted.** Did you include `cudaMalloc` and the
host-device transfers in your timing? The first CUDA call in a process pays
~100 ms of context initialisation. Warm up.

---

## 9. Exercises

Written work first, then `02-vector-add` and `02-saxpy`.

### 2.1 - Index arithmetic on paper

`gridDim = 4`, `blockDim = 64`.

(a) Global index of `threadIdx.x = 10` in `blockIdx.x = 2`?
(b) Which block and thread owns global index 200?
(c) How many warps in this grid? Which warp holds global index 200?
(d) With `n = 250`, how many threads have no work? Which block are they in?

### 2.2 - 2D flattening

A 1024 x 768 image, row-major, `dim3 block(16,16)`.

(a) Grid dimensions?
(b) For `blockIdx = (3,5)`, `threadIdx = (2,7)`, the pixel (x,y)?
(c) Its flat offset in a row-major array?
(d) Warp 0 of that block covers which 32 pixels? Are their flat offsets
    consecutive?
(e) Repeat (d) for `dim3 block(8,32)`. Which block shape would you pick for a
    row-major image, and why?

### 2.3 - Grid sizing

`n = 1,000,000`, `block = 256`.

(a) Blocks for one-thread-per-element?
(b) On a 108-SM GPU with 8 blocks resident per SM, how many *waves*?
(c) The last wave is partial. What fraction of the GPU is idle during it?
(d) Design a grid-stride launch with no tail effect. What grid size, and what
    does each thread do?

### 2.4 - Predict the vector add

Before writing it: on your GPU, for `n = 64M` fp32.

(a) Bytes moved (count writes carefully -- re-read Chapter 01 section 2).
(b) Predicted time at your *measured* copy ceiling.
(c) Predicted achieved GB/s as the harness will report it.
(d) Predicted GFLOP/s, and what fraction of peak that is.

Write these down. Then do the exercise. Then compare.

---

## Exercises to run

- [`02-vector-add`](../exercises/02-vector-add) - the canonical first kernel,
  with a real performance bar.
- [`02-saxpy`](../exercises/02-saxpy) - same shape, different traffic. The
  interesting part is explaining why the GB/s differ.

---

## What you should now believe

- Host and device memory are separate and transfers between them are slow.
- `blockIdx.x * blockDim.x + threadIdx.x` and the ceil-divide are muscle memory.
- Every kernel over a non-multiple size needs a bounds guard.
- Launches are async; errors surface late; check twice.
- x is the fastest-varying dimension and that fact governs memory layout.

Next: [Chapter 03 - Memory hierarchy and coalescing](03-memory-hierarchy.md),
where the performance actually comes from.
