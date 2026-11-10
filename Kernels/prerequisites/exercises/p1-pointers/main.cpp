// P1 - pointers
//
// Every function here has the shape of a CUDA kernel: it is handed addresses
// and a length, and it writes its answer through a pointer. Nothing returns an
// array, because nothing can.
//
// Chapter: P2

#include <cstdio>
#include <vector>

#include "check.h"

// ---------------------------------------------------------------- 1. swap
// Exchange the two integers the caller owns. Not copies of them.
void swap_ints(int* a, int* b) {
    // TODO
    (void)a;
    (void)b;
}

// ------------------------------------------------------------- 2. read it
// Return the value stored at the given address.
float read_through(const float* p) {
    // TODO
    (void)p;
    return 0.0f;
}

// ------------------------------------------------------------ 3. write it
// Store `value` at the given address.
void write_through(float* p, float value) {
    // TODO
    (void)p;
    (void)value;
}

// ----------------------------------------------------------- 4. fill
// Set n consecutive floats starting at `out` to `value`.
void fill(float* out, int n, float value) {
    // TODO
    (void)out;
    (void)n;
    (void)value;
}

// ------------------------------------------------------------- 5. sum
// Sum n floats. Accumulate in double -- P4 section 5 explains why, and the
// test here is large enough that accumulating in float will fail.
double sum(const float* data, int n) {
    // TODO
    (void)data;
    (void)n;
    return 0.0;
}

// ------------------------------------------------------- 6. elementwise
// out[i] = a[i] + b[i], for n elements.
// This is 02-vector-add with the parallelism taken out.
void add_arrays(const float* a, const float* b, float* out, int n) {
    // TODO
    (void)a;
    (void)b;
    (void)out;
    (void)n;
}

// ------------------------------------------------------- 7. two outputs
// A function cannot return two things. It can be handed two addresses.
// Write the smallest element to *out_min and the largest to *out_max.
void min_max(const float* data, int n, float* out_min, float* out_max) {
    // TODO
    (void)data;
    (void)n;
    (void)out_min;
    (void)out_max;
}

// ------------------------------------------------- 8. pointer arithmetic
// Return a pointer to element k of the array, WITHOUT using [].
// One expression. This is the whole of P2 section 4.
const float* element_at(const float* base, int k) {
    // TODO
    (void)base;
    (void)k;
    return nullptr;
}

// ------------------------------------------------------ 9. a pointer out
// Set *out to point at the LARGEST element of the array.
// Note the two stars: the function must change the caller's pointer, so it
// needs the address of that pointer. This is exactly cudaMalloc's signature
// shape, and it is the only genuinely confusing one here.
void find_largest(const float* data, int n, const float** out) {
    // TODO
    (void)data;
    (void)n;
    (void)out;
}

int main() {
    check::Suite s("p1-pointers");

    // 1
    int x = 3, y = 7;
    swap_ints(&x, &y);
    s.eq_int("swap_ints: x", x, 7);
    s.eq_int("swap_ints: y", y, 3);
    int z = 5;
    swap_ints(&z, &z);
    s.eq_int("swap_ints: aliased (a == b)", z, 5);

    // 2, 3
    float v = 2.5f;
    s.eq_double("read_through", read_through(&v), 2.5);
    write_through(&v, 9.5f);
    s.eq_double("write_through changed the caller's variable", v, 9.5);

    // 4
    std::vector<float> buf(10, -1.0f);
    fill(buf.data(), 10, 4.0f);
    bool all_four = true;
    for (int i = 0; i < 10; i++) all_four = all_four && buf[i] == 4.0f;
    s.ok("fill: all ten elements set", all_four);

    fill(buf.data(), 0, 99.0f);
    s.ok("fill: n = 0 writes nothing", buf[0] == 4.0f);

    // 5
    std::vector<float> ramp(1000);
    for (int i = 0; i < 1000; i++) ramp[i] = float(i);
    s.eq_double("sum of 0..999", sum(ramp.data(), 1000), 499500.0, 1e-6);
    s.eq_double("sum of an empty range", sum(ramp.data(), 0), 0.0);

    // 20 million is past 2^24, the point where a float accumulator stops being
    // able to count. If this check reports exactly 16777216, you now know what
    // P4 section 4 was talking about.
    const int big_n = 20000000;
    std::vector<float> ones(big_n, 1.0f);
    s.eq_double("sum of 2e7 ones (needs a wide accumulator)",
                sum(ones.data(), big_n), double(big_n), 1e-6);

    // 6
    std::vector<float> a = {1, 2, 3, 4}, b = {10, 20, 30, 40}, out(4, 0.0f);
    add_arrays(a.data(), b.data(), out.data(), 4);
    s.eq_array("add_arrays", {out[0], out[1], out[2], out[3]}, {11, 22, 33, 44});

    // 7
    std::vector<float> mixed = {3.0f, -1.0f, 7.0f, 0.0f, 7.0f, -9.0f};
    float lo = 0.0f, hi = 0.0f;
    min_max(mixed.data(), 6, &lo, &hi);
    s.eq_double("min_max: min", lo, -9.0);
    s.eq_double("min_max: max", hi, 7.0);

    std::vector<float> all_neg = {-5.0f, -2.0f, -8.0f};
    min_max(all_neg.data(), 3, &lo, &hi);
    s.eq_double("min_max: max of all-negative data", hi, -2.0);

    // 8
    const float* p3 = element_at(ramp.data(), 3);
    s.ok("element_at: returns non-null", p3 != nullptr);
    if (p3) {
        s.eq_double("element_at: correct value", *p3, 3.0);
        s.eq_int("element_at: correct address offset",
                 (long long)(p3 - ramp.data()), 3);
    }

    // 9
    const float* big = nullptr;
    find_largest(mixed.data(), 6, &big);
    s.ok("find_largest: set the pointer", big != nullptr);
    if (big) {
        s.eq_double("find_largest: points at the max value", *big, 7.0);
        s.eq_int("find_largest: points at the FIRST such element",
                 (long long)(big - mixed.data()), 2);
    }

    s.note("Crashing? `make debug` names the exact line.");
    return s.finish();
}
