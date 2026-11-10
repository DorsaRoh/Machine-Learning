// P0 - hello
//
// The smallest possible exercise: prove your compiler works, print something,
// and compute the sizes you will be using for the rest of the course.
//
// Chapters: P0, P1

#include <cstdio>

#include "check.h"

int main() {
    check::Suite s("p0-hello");

    // ---- 1. Make it print ------------------------------------------------
    // TODO: print the word  hello  followed by a newline.
    // (This one is not checked automatically. Look at your terminal.)

    // ---- 2. Sizes --------------------------------------------------------
    // TODO: replace each 0 with the number of BYTES that type occupies.
    // Do NOT use sizeof() for these four. Answer from memory (P0 section 1),
    // then check yourself with sizeof afterwards -- the point is to know.
    size_t bytes_in_a_float = 0;
    size_t bytes_in_a_double = 0;
    size_t bytes_in_an_int = 0;
    size_t bytes_in_a_char = 0;

    s.eq_size("a float is _ bytes", bytes_in_a_float, sizeof(float));
    s.eq_size("a double is _ bytes", bytes_in_a_double, sizeof(double));
    s.eq_size("an int is _ bytes", bytes_in_an_int, sizeof(int));
    s.eq_size("a char is _ bytes", bytes_in_a_char, sizeof(char));

    // ---- 3. Arithmetic you will do a thousand times -----------------------
    // TODO: how many bytes does an array of 1,000,000 floats occupy?
    size_t bytes_for_a_million_floats = 0;
    s.eq_size("1e6 floats in bytes", bytes_for_a_million_floats, 4000000);

    // TODO: how many floats fit in exactly 1 gigabyte (1e9 bytes)?
    size_t floats_in_a_gigabyte = 0;
    s.eq_size("floats per GB", floats_in_a_gigabyte, 250000000);

    // TODO: a 7-billion-parameter model, 2 bytes per parameter (fp16).
    // How many bytes? (You will meet this exact number in Chapter 21.)
    size_t bytes_for_7b_fp16 = 0;
    s.eq_size("7B params at fp16", bytes_for_7b_fp16, 14000000000ULL);

    // ---- 4. Integer division ---------------------------------------------
    // TODO: implement ceiling division: a / b, rounded UP.
    // No floating point, no if-statements needed. P1 section 2.
    //
    // This is how every CUDA launch in the course computes its block count,
    // so getting it exactly right matters more than it looks.
    auto ceil_div = [](int a, int b) -> int {
        (void)a;
        (void)b;
        return 0;  // TODO
    };

    s.eq_int("ceil_div(0, 4)", ceil_div(0, 4), 0);
    s.eq_int("ceil_div(1, 4)", ceil_div(1, 4), 1);
    s.eq_int("ceil_div(4, 4)", ceil_div(4, 4), 1);
    s.eq_int("ceil_div(5, 4)", ceil_div(5, 4), 2);
    s.eq_int("ceil_div(1000, 256)", ceil_div(1000, 256), 4);
    s.eq_int("ceil_div(1024, 256)", ceil_div(1024, 256), 4);
    s.eq_int("ceil_div(1025, 256)", ceil_div(1025, 256), 5);

    // ---- 5. The point of all this ----------------------------------------
    // A kernel launching ceil_div(n, 256) blocks of 256 threads covers n
    // elements. Launching n / 256 blocks does not.
    //
    // TODO: how many of 1000 elements would a grid of (1000 / 256) blocks of
    // 256 threads fail to process?
    int elements_missed = -1;
    s.eq_int("elements missed by n/256 blocks", elements_missed, 232);

    s.note("Checks tell you WHAT is wrong, never how to fix it. That is the deal.");
    return s.finish();
}
