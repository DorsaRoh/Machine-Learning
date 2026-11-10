# 06 - transpose

**Chapter:** [06](../../chapters/06-shared-memory-tiling.md)

## Task

Three implementations of the same 8192 x 8192 fp32 transpose: naive, tiled, and
tiled with padded shared memory. The best must reach 85% of your copy ceiling.

## Predict first

```
          predicted GB/s   predicted % of ceiling   bottleneck
v1 naive      ______              ______            ______
v2 tiled      ______              ______            ______
v3 padded     ______              ______            ______

predicted speedup v1 -> v2 : ______x   mechanism: ______
predicted speedup v2 -> v3 : ______x   mechanism: ______
```

Two different mechanisms, two different numbers. If you predict them both
within 30% you have understood Chapter 06.

## Run

```bash
make run
make run ARGS=4096
make run ARGS=16384    # if you have the memory
```

## Pass bar

- Correctness: all three exact.
- Performance: best version >= 85% of `copy_ceiling_gbs()`.

## The gotcha

v2 and v3 have an index trap. The block that *reads* tile `(bx, by)` must
*write* tile `(by, bx)`. If you write back to the same block coordinates you
will produce a result that is locally transposed but globally wrong -- and on a
symmetric test matrix it might even look correct. The harness uses random data
specifically so this bug cannot hide.

## After it passes

1. **Verify the conflict story.**
   ```bash
   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./exercise
   ```
   v2 should show a large load-conflict count and v3 essentially zero. If v2's
   conflicts are on the *store* instead of the load, you built the tile the
   other way round -- which is fine, but make sure you know which.

2. **Why not exactly 100%?** Even v3 falls short of a pure copy. Find the
   remaining cost. Candidates: the barrier, the extra shared traffic, partial
   tiles at the edges, and the fact that reads and writes now hit different
   DRAM pages. Rank them by size.

3. **Tile size sweep.** Try TILE = 16, 32, 64 with BLOCK_ROWS adjusted. Compute
   the shared footprint and occupancy for each before running. Does the winner
   match your prediction?

4. **BLOCK_ROWS.** The default 32x8 gives each thread 4 elements. Try 32x32
   (one element each) and 32x4 (eight each). Explain the result using ILP and
   Little's Law, not by trial and error.

5. **Write a swizzled v4.** No padding, XOR swizzle instead (Chapter 06
   exercise 6.3). Same speed as v3? It should be. Now you have the technique
   you need for Chapter 13, where padding is not an option.

6. **Non-square.** Make it work for `M x N` with `M != N`. What breaks? This is
   not busywork -- every real transpose in a transformer is non-square.

<details>
<summary>Hints</summary>

- In the load phase, which thread index varies fastest, and which memory index
  does it drive? Same question for the store phase.
- The tile is 32 wide. A warp is 32 lanes. How many rows of the tile does one
  warp cover per iteration of your load loop?
- For the shared read `tile[threadIdx.x][threadIdx.y]`: compute the bank for
  lane 0, lane 1, lane 2. Are they different?
- With `tile[32][33]`, what is the float offset of `tile[i][0]`? Mod 32?
- Output tile origin: if your block read rows `by*32..` and columns `bx*32..`,
  which rows and columns of the output does that data belong to?

</details>
