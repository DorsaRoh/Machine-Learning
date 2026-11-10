# P6 - The toolchain

> Half of learning a systems language is learning to read what the tools say
> when you get it wrong. A compiler error is not a rejection; it is the most
> precise code review you will ever receive, and it arrives in under a second.

---

## 1. What happens when you compile

```bash
c++ -std=c++17 -O2 -Wall main.cpp -o exercise
```

Four stages, and knowing which one failed tells you what kind of mistake you
made:

| Stage | Does | Fails when |
|---|---|---|
| Preprocessor | pastes in `#include`s, expands `#define`s | a header is missing |
| Compiler | your code → machine code, one file at a time | syntax, types, undeclared names |
| Assembler | textual machine code → object file `.o` | essentially never |
| Linker | combines `.o` files and libraries → executable | a function was declared but never defined |

The practical distinction is **compiler error** versus **linker error**.

A compiler error names a file and a line. A linker error names a *symbol* and
no line at all:

```
Undefined symbols for architecture arm64:
  "add(float, float)", referenced from: _main in main.o
```

That means: you declared `add`, you called it, and nobody ever wrote the body —
or you wrote it in a file that was not compiled in. Line numbers will not help;
go looking for a missing definition.

The CUDA version is the same picture with `nvcc` in place of `c++`, splitting
your file into host code (handed to the system compiler) and device code
(compiled to PTX and SASS). `-lcublas` adds a library at the link stage, which
is why `11-gemm-ladder`'s Makefile has it.

---

## 2. Flags that matter

```
-std=c++17     which version of the language
-O0            no optimisation. Fast builds, slow code, honest debugging.
-O2 / -O3      optimise. What you measure with.
-g             include debug information (line numbers in crashes)
-Wall -Wextra  turn on warnings
-I<dir>        also look in <dir> for headers
-l<name>       link against lib<name>
-o <file>      name the output
```

**Never benchmark a `-O0` build.** It can be ten times slower and the ratios
between versions are meaningless. Equally, never debug a `-O3` build without
`-g`: the optimiser reorders and inlines so aggressively that the line numbers
in a stack trace stop corresponding to anything.

**Treat warnings as errors in your head.** `-Wall` catches uninitialised
variables, `=` instead of `==`, `printf` format mismatches, and unused results
— which is to say, it catches a solid third of the bugs in this chapter's
exercises before you run anything.

---

## 3. `make`, enough to use it

`make` rebuilds only what changed. A Makefile is a list of rules:

```make
exercise: main.cpp kernel.h
	c++ -std=c++17 -O2 main.cpp -o exercise
```

Read as: "to build `exercise`, which depends on `main.cpp` and `kernel.h`, run
this command." If `exercise` is newer than both dependencies, `make` does
nothing.

**The recipe line must start with a TAB, not spaces.** This is make's most
famous and least forgivable quirk. If you get `Makefile:3: *** missing
separator. Stop.`, that is what happened.

Targets that are commands rather than files are marked `.PHONY`:

```make
.PHONY: run clean
run: exercise
	./exercise
clean:
	rm -f exercise
```

Variables use `:=` (evaluate now) or `=` (evaluate at use), and are referenced
with `$(NAME)`:

```make
CXXFLAGS := -std=c++17 -O2 -Wall
exercise: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@      # $@ is the target name
```

You never need to write one for this course — every exercise has a two-line
Makefile that includes a shared one. But you should be able to open
`prerequisites/common/common.mk` or `common/common.mk` and see what `make run`
is actually going to do.

Useful in practice:

```bash
make            # build
make run        # build and run
make clean      # delete the outputs
make -n run     # print the commands WITHOUT running them. Very useful.
make ARGS="4096 1024" run     # pass arguments through
```

---

## 4. Reading an error message

The rule: **fix the first error, then recompile.** One mistake routinely
produces twenty messages, and nineteen of them are the compiler flailing after
it lost its place. Never start from the bottom.

Error types, with what they actually mean:

**`expected ';' before ...`** — the semicolon is missing on the line *above*
the one reported. The compiler only noticed when it hit the next token.

**`use of undeclared identifier 'foo'`** — typo, or a missing `#include`, or
you used a variable outside the braces where it was declared.

**`no matching function for call to 'foo'`** — it exists, but not with those
argument types. The compiler then lists the candidates it rejected and why;
that list is the answer, even though it is ugly.

**`invalid conversion from 'int*' to 'int'`** — you have a pointer where a
value belongs, or the reverse. Missing `*` or missing `&`. P2 section 1.

**`undefined reference to ...` / `Undefined symbols`** — linker, section 1.

**Template errors** are famously enormous. The technique: find the *first*
line, and the line mentioning *your* file. The 200 lines of instantiation
backtrace are context, not content.

---

## 5. When it compiles but misbehaves

Three tools, in the order you should reach for them.

**`printf`.** Unfashionable and extremely effective. Print the index, the
address, the value, and the loop bounds. In this course it is also the only
debugger available inside a GPU kernel, so you may as well be good at it.

The discipline that makes it work: **shrink the problem first.** Run with 8
elements instead of 8 million and print everything. Almost every indexing bug
is visible at n=8 and invisible at n=8,000,000. The main course's "when you are
stuck" list says the same thing at step 5.

**The sanitizers.** This is the big one for P2-style bugs:

```bash
make debug
```

which compiles with `-fsanitize=address,undefined -O0 -g` and runs. It reports:

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
READ of size 4 at 0x... thread T0
    #0 0x... in sum(float const*, int) main.cpp:14
0x... is located 0 bytes to the right of 400-byte region
```

Read that as: at `main.cpp:14` you read 4 bytes immediately past the end of a
400-byte (= 100 float) allocation. That is `a[100]` on a 100-element array, and
it just told you the exact line.

AddressSanitizer catches all four bugs from P2 section 7. UndefinedBehavior-
Sanitizer catches signed overflow, bad shifts, and misaligned access. Together
they cost you a slower binary and save you entire evenings.

**The direct equivalent in the main course is `compute-sanitizer`**, which you
will meet in Chapter 18 and which finds the same families of bug on the GPU
plus data races. Building the habit now means that chapter is familiar rather
than new.

**A debugger**, if you want one: `lldb ./exercise` on macOS, `gdb` on Linux.
`run`, then `bt` for a backtrace when it crashes, `p variable` to inspect.
Worth half an hour of your time, but honestly the sanitizers plus `printf`
cover almost everything at this level.

---

## 6. Exercises

### P6.1 - Cause each error deliberately

For each, write the smallest program that produces it, and record the exact
message:

(a) Missing semicolon.
(b) Undeclared identifier.
(c) Missing `#include`.
(d) Wrong argument type to a function.
(e) A pointer/value mismatch (missing `*` or `&`).
(f) A linker error: declare a function, call it, never define it.
(g) A warning that `-Wall` catches but that compiles fine anyway.

Keep the list. Next time you see one of these for real you will recognise it in
two seconds instead of two minutes.

### P6.2 - One mistake, twenty errors

(a) Delete a single closing brace somewhere in the middle of a 50-line file.
(b) Compile. How many errors?
(c) Does the first one point at the real problem?
(d) Fix only the first error. Recompile. How many now?
(e) State the rule you just derived.

### P6.3 - The sanitizers earn their keep

Take your P2.5 program (the four deliberate bugs).

(a) Run each under `make run`. Record what happens — including "nothing".
(b) Run each under `make debug`. Record the message.
(c) Which bugs were completely silent in (a)?
(d) For the out-of-bounds one, change how far past the end you read: 1 element,
    10, 1000. Does the `make run` behaviour change? Does `make debug`?
(e) Write one sentence about why "it ran fine" is not evidence of correctness.

### P6.4 - Read the build

(a) `cd prerequisites/exercises/p0-hello && make -n run`. What command does it
    actually run?
(b) Open `prerequisites/common/common.mk` and find where those flags come from.
(c) What does `make debug` change, and why both `-O0` and the sanitizer flags?
(d) Now do the same for `Kernels/common/common.mk`. What does `make sanitize`
    run there, and what is it the GPU analogue of?

### P6.5 - Optimisation is not free of consequences

(a) Write a loop summing 100 million floats. Time it built with `-O0`, `-O1`,
    `-O2`, `-O3`.
(b) What ratio between `-O0` and `-O2`?
(c) Add `-g` to the `-O2` build. Does the time change? Does anything?
(d) Why is "never benchmark `-O0`, never debug `-O3` without `-g`" the rule?

---

## What you should now believe

- Compiling has stages, and compiler errors versus linker errors tell you
  different things. Linker errors have no line number because they are not
  about a line.
- Fix the first error only, then recompile.
- `-O0` for debugging, `-O2`/`-O3` for measuring, `-g` always, `-Wall` always.
- `make` rebuilds what changed; `make -n` shows you what it would do.
- Shrink the input before you debug. Most indexing bugs are visible at n=8.
- The sanitizers find the bugs that are silent at runtime, and they are the
  laptop version of `compute-sanitizer`.

Next: [P7 - The math you need](p7-the-math.md).
