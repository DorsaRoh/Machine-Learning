# P0 - What a computer actually is

> You can write code for years without knowing this. You cannot write *fast*
> code without it, because every optimisation in this course is a statement
> about where data is and how far it has to travel.

---

## 1. Everything is a number, and every number is bits

A **bit** is one of two things: 0 or 1. That is the whole of it. A wire that is
either on or off.

Eight bits make a **byte**. Eight bits have 2^8 = 256 possible patterns, so one
byte can represent 256 distinct things — the numbers 0 to 255, say, or 256
different letters.

That is the only representation a computer has. Not "numbers and also text and
also images" — just bits, interpreted differently depending on what the program
decides they mean:

```
bit pattern 01000001
  as a number  -> 65
  as a letter  -> 'A'   (because someone decided 65 means 'A')
  as part of a float -> some fragment of an exponent
```

**Nothing in memory carries a label saying what it is.** The bits for the
number 65 and the bits for the letter 'A' are identical. The only thing that
makes one an `int` and the other a `char` is the type you told the compiler.
This is why C lets you reinterpret memory, and why doing it wrong produces
garbage rather than an error.

Sizes you should memorise, because you will compute with them constantly:

| Type | Bytes | Bits | Range or precision |
|---|---|---|---|
| `char` | 1 | 8 | -128 to 127 |
| `short` | 2 | 16 | -32768 to 32767 |
| `int` | 4 | 32 | about -2.1 billion to 2.1 billion |
| `long long` | 8 | 64 | about -9.2e18 to 9.2e18 |
| `float` | 4 | 32 | ~7 decimal digits |
| `double` | 8 | 64 | ~16 decimal digits |
| `half` (GPU) | 2 | 16 | ~3 decimal digits |

**A `float` is 4 bytes.** You will use that fact in nearly every calculation in
this course. A million floats is 4 megabytes. Burn it in.

---

## 2. Memory is one enormous array of bytes

Here is the model that matters. Your computer's RAM is a single, gigantic,
numbered sequence of bytes:

```
address:   0      1      2      3      4      5      6      7     ...
contents: [ 01 ] [ 00 ] [ 00 ] [ 00 ] [ 41 ] [ 42 ] [ 43 ] [ 00 ] ...
```

Every byte has an **address** — its position in that sequence. Addresses are
just integers, counting from zero. 16 GB of RAM means addresses 0 through
roughly 17,179,869,183.

When you write:

```c
int x = 1;
```

the computer picks four consecutive bytes somewhere — say addresses 1000, 1001,
1002, 1003 — and writes the bit pattern for 1 into them. The name `x` is a
convenience for you; the machine only ever deals in the address.

**This is the single most important idea in this part.** A variable is a *name
for an address*. Once you believe that, pointers stop being mysterious: a
pointer is just a variable whose contents happen to be an address.

Things that follow immediately, and that you will use in Chapter 03:

- Data that is **next to each other in memory** has addresses that differ by a
  small number.
- An array of 100 floats occupies 400 consecutive bytes. Element `i` starts at
  `base + 4*i`.
- "Is this access contiguous?" is a question about whether consecutive
  operations touch consecutive addresses. It has a precise answer, and that
  answer determines performance.

---

## 3. The processor is separate from the memory, and that is the whole problem

A CPU (and a GPU) can only compute on data that is *inside* it, in tiny storage
slots called **registers**. Anything in RAM has to be fetched first:

```
          fetch
RAM  ------------->  register  ---> arithmetic unit
     <-------------
          store
```

So every computation is really three steps: load, compute, store.

Here is the fact that this entire course is built on. Those three steps do not
cost the same:

| Operation | Roughly how long |
|---|---|
| Add two numbers already in registers | 1 unit of time |
| Fetch a number from RAM | 200-400 units of time |

Reading a number from memory costs hundreds of times more than doing arithmetic
on it. **Computers are not slow at maths. They are slow at fetching.**

Say that back to yourself, because almost all performance work — on a CPU, on a
GPU, in this entire course — is a consequence of it. When Chapter 01 tells you
a kernel is "memory-bound", it means the arithmetic is free and you are paying
entirely for the fetching.

---

## 4. Cache: the fix, and why it shapes your code

Since RAM is slow, processors keep small, fast copies of recently used data
nearby. These are **caches**, and they come in levels:

```
registers   <1 unit      a few hundred bytes    inside the core
L1 cache     ~4 units    ~32-128 KB             per core
L2 cache    ~15 units    ~1-50 MB               shared
RAM        ~250 units    8-128 GB               off-chip
```

Smaller and faster as you go up. Bigger and slower as you go down. This is the
**memory hierarchy**, and the GPU version in Chapter 03 is the same picture
with different numbers.

Two rules govern what gets cached, and they both have direct consequences for
how you write loops:

**Rule 1: memory moves in chunks, not bytes.** When you ask for one byte, the
hardware fetches an entire **cache line** — typically 64 bytes on a CPU — and
keeps it. If you then ask for the next byte, it is already there and costs
almost nothing.

So this loop is fast:

```c
for (int i = 0; i < n; i++) sum += a[i];     // consecutive addresses
```

and this loop, doing exactly the same number of additions, can be ten times
slower:

```c
for (int i = 0; i < n; i += 16) sum += a[i];  // skips 64 bytes each time
```

The second one fetches a full 64-byte line and uses 4 bytes of it. It throws
away 94% of every fetch.

**That is the entire idea behind "coalescing" in Chapter 03.** You already
understand it; the GPU version just has 32 threads doing it at once.

**Rule 2: recently used data stays.** If you touch something twice in quick
succession, the second touch is nearly free. If you touch a million other
things in between, it has been evicted and you pay full price again.

This is why **tiling** works — Chapter 06, and the heart of every fast matrix
multiply. You reorganise a loop so that the data you reuse is still in fast
memory when you come back to it. Same arithmetic, same answer, different order,
ten times faster.

---

## 5. What a program actually is

A compiled program is a sequence of extremely simple instructions:

```
load the value at address 1000 into register 1
load the value at address 1004 into register 2
add register 1 and register 2, put the result in register 3
store register 3 at address 1008
```

That is the machine code — the SASS you will read in Chapter 16. Four
instructions for what you wrote as `c = a + b;`.

The **compiler** is the program that turns your C++ into that. Its job is to
preserve what your code *means* while being free to change how it is done:
reorder instructions, keep values in registers, eliminate work it can prove is
unnecessary, unroll loops.

Two consequences worth knowing now:

- **The compiler is on your side** and is extremely good. Most "clever" tricks
  you might write are things it already does. In Chapter 16 you will check what
  it actually did rather than guessing.
- **The compiler assumes your code is correct.** If you read off the end of an
  array, it does not stop you; it generates instructions that read whatever
  bytes are there. This is why C is fast and why it will let you cause chaos.
  `make debug` and the sanitizers exist for exactly this.

---

## 6. Where the GPU fits

You will meet this properly in Chapter 00. For now, one sentence:

**A GPU is the same picture — processor, registers, cache, memory — but with
thousands of tiny processors instead of eight big ones, and a memory system
built for moving enormous contiguous blocks rather than for responding
quickly.**

Everything in this chapter still applies there. The numbers change. The ratios
get more extreme. The conclusions get sharper.

---

## 7. Exercises

Pencil and paper. No compiler.

### P0.1 - Sizes

(a) How many bytes is an array of 1000 `float`s? Of 1000 `double`s?
(b) How many floats fit in 1 MB? In 1 GB?
(c) A matrix of 1024 x 1024 floats — how many bytes? How many megabytes?
(d) An image, 1920 x 1080 pixels, 3 bytes per pixel. How many MB?
(e) A neural network with 7 billion parameters, each a 2-byte `half`. How many
    GB? (You will use this exact number in Chapter 21.)

### P0.2 - Addresses

An array of floats starts at address 2000.

(a) What address is element 0 at? Element 1? Element 10?
(b) What is the address of element `i`, as a formula?
(c) If it were an array of `double`s instead, what changes?
(d) Element 5 is at address 2020. Someone reads 4 bytes from address 2018. What
    do they get? Is this an error?

### P0.3 - Cache arithmetic

Cache line is 64 bytes. Array of floats.

(a) How many floats fit in one cache line?
(b) Looping `i = 0, 1, 2, ...` — how many array elements do you get per line
    fetched?
(c) Looping `i = 0, 16, 32, ...` — same question.
(d) How much slower do you expect (c) to be than (b), and why is the real
    answer usually *less* extreme than the arithmetic suggests?
(e) Looping `i = 0, 1000, 2000, ...` over a 10 MB array — what fraction of each
    fetched line do you use?

### P0.4 - The ratio

Assume an add takes 1 unit and a RAM fetch takes 250 units.

(a) `sum += a[i]` for a million elements, with every fetch missing cache: total
    time?
(b) Same, but 15 of every 16 fetches hit in cache (because of lines, from P0.3):
    total time?
(c) What fraction of time in (b) is spent on arithmetic?
(d) If you made the arithmetic twice as fast, how much faster is the loop?
(e) If you made the fetching twice as fast, how much faster is the loop?
(f) Which of those two would you work on, and what does that tell you about
    where performance work actually happens?

---

## What you should now believe

- Everything is bits; types are an interpretation you impose, not a property of
  the memory.
- Memory is one enormous numbered array of bytes, and a variable is a name for
  an address.
- Fetching from RAM costs hundreds of times what arithmetic costs. Computers
  are slow at fetching, not at maths.
- Memory moves in cache lines, so using all of a line is free and using one
  byte of it wastes the rest. This is coalescing, before anyone calls it that.
- Reusing data while it is still in fast memory is why tiling works.
- `float` is 4 bytes. A million floats is 4 MB.

Next: [P1 - C essentials](p1-c-essentials.md), where you compile and run
something.
