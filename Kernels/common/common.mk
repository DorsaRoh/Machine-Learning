# Shared build rules for every exercise. Each exercise Makefile is three lines:
#
#   EXERCISE := 02-vector-add
#   include ../../common/common.mk
#
# Targets:
#   make            build
#   make run        build + run the gate
#   make sanitize   run under compute-sanitizer (races, OOB, leaks)
#   make profile    run under Nsight Compute, speed-of-light + stall sections
#   make sass       dump the generated machine code for your kernel
#   make clean

NVCC    ?= nvcc
COMMON  := $(dir $(lastword $(MAKEFILE_LIST)))
BIN     := exercise

# -arch=native (CUDA >= 11.5) compiles for whatever GPU you actually rented.
# Older toolkits fall back to Volta, which runs everywhere from sm_70 up.
ARCHFLAG := $(shell $(NVCC) -arch=native -E -x cu /dev/null > /dev/null 2>&1 \
              && echo -arch=native || echo -arch=sm_70)

# -lineinfo keeps the SASS mapped back to your source lines, which is what
# makes Nsight Compute's per-line stall attribution readable. It costs nothing.
NVCCFLAGS ?= -O3 -std=c++17 $(ARCHFLAG) -lineinfo -I$(COMMON) \
             --expt-relaxed-constexpr -Xcompiler -Wall

SRCS := main.cu
DEPS := $(wildcard *.cu *.cuh) $(COMMON)bench.cuh

.PHONY: all run sanitize profile sass clean

all: $(BIN)

$(BIN): $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@ $(LDFLAGS)

run: $(BIN)
	./$(BIN) $(ARGS)

sanitize: $(BIN)
	compute-sanitizer --tool memcheck ./$(BIN) $(ARGS)
	compute-sanitizer --tool racecheck ./$(BIN) $(ARGS)

profile: $(BIN)
	ncu --set full --print-summary per-kernel ./$(BIN) $(ARGS)

sass: $(BIN)
	cuobjdump -sass $(BIN) | less

clean:
	rm -f $(BIN) *.o
