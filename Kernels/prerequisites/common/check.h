// Tiny self-checking harness for the prerequisite exercises.
//
// Deliberately much simpler than common/bench.cuh: there is no GPU here, no
// timing, no performance gate. All we need is "did you get the right answer,
// and if not, what exactly was wrong".
//
// Everything in prerequisites/ compiles with a plain C++ compiler on any
// machine, including your Mac. No CUDA, no GPU, no rented instance.

#ifndef PREREQ_CHECK_H
#define PREREQ_CHECK_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace check {

class Suite {
   public:
    explicit Suite(std::string name) : name_(std::move(name)) {
        std::printf("\n=== %s ===\n\n", name_.c_str());
    }

    // ---- the primitives -------------------------------------------------

    void ok(const char* what, bool condition, const char* detail = nullptr) {
        ++total_;
        if (condition) {
            ++passed_;
            std::printf("  [pass] %s\n", what);
        } else {
            std::printf("  [FAIL] %s\n", what);
            if (detail) std::printf("         %s\n", detail);
        }
    }

    void eq_int(const char* what, long long got, long long want) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "got %lld, expected %lld", got, want);
        ok(what, got == want, buf);
    }

    void eq_size(const char* what, size_t got, size_t want) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "got %zu, expected %zu", got, want);
        ok(what, got == want, buf);
    }

    void eq_double(const char* what, double got, double want, double tol = 1e-9) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "got %.12g, expected %.12g (tol %g)", got,
                      want, tol);
        ok(what, std::fabs(got - want) <= tol, buf);
    }

    void eq_str(const char* what, const char* got, const char* want) {
        char buf[300];
        std::snprintf(buf, sizeof buf, "got \"%s\", expected \"%s\"",
                      got ? got : "(null)", want);
        ok(what, got && std::strcmp(got, want) == 0, buf);
    }

    // Compares a whole array and reports the FIRST disagreement, because that
    // is the one that tells you what you did wrong. The tenth is noise.
    void eq_array(const char* what, const std::vector<double>& got,
                  const std::vector<double>& want, double tol = 1e-9) {
        char buf[300];
        if (got.size() != want.size()) {
            std::snprintf(buf, sizeof buf, "length %zu, expected %zu", got.size(),
                          want.size());
            ok(what, false, buf);
            return;
        }
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::isnan(got[i]) || std::fabs(got[i] - want[i]) > tol) {
                std::snprintf(buf, sizeof buf,
                              "first difference at index %zu: got %.12g, expected %.12g",
                              i, got[i], want[i]);
                ok(what, false, buf);
                return;
            }
        }
        ok(what, true);
    }

    // For a 2D result, the index alone is not enough: you need the (row, col)
    // to see that you transposed something.
    void eq_matrix(const char* what, const std::vector<double>& got,
                   const std::vector<double>& want, int rows, int cols,
                   double tol = 1e-9) {
        char buf[300];
        if (got.size() != want.size()) {
            std::snprintf(buf, sizeof buf, "length %zu, expected %zu", got.size(),
                          want.size());
            ok(what, false, buf);
            return;
        }
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                size_t i = size_t(r) * cols + c;
                if (std::isnan(got[i]) || std::fabs(got[i] - want[i]) > tol) {
                    std::snprintf(buf, sizeof buf,
                                  "first difference at row %d, col %d (flat index %zu): "
                                  "got %.12g, expected %.12g",
                                  r, c, i, got[i], want[i]);
                    ok(what, false, buf);
                    return;
                }
            }
        }
        ok(what, true);
    }

    void note(const char* fmt) { std::printf("\n  %s\n\n", fmt); }

    int finish() {
        std::printf("\n  %d / %d checks passed\n", passed_, total_);
        std::printf("-------------------------------------------------\n");
        bool all = passed_ == total_ && total_ > 0;
        std::printf("%s: %s\n\n", name_.c_str(), all ? "PASSED" : "NOT YET");
        return all ? 0 : 1;
    }

   private:
    std::string name_;
    int total_ = 0;
    int passed_ = 0;
};

}  // namespace check

#endif  // PREREQ_CHECK_H
