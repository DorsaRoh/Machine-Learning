EXERCISE := 25-allreduce
NVCC ?= nvcc

# NCCL is the baseline, but the exercise is still worth doing without it.
NCCL_OK := $(shell echo '\#include <nccl.h>' | $(NVCC) -E -x cu - >/dev/null 2>&1 && echo 1)
ifeq ($(NCCL_OK),1)
  LDFLAGS := -lnccl
  EXTRA   := -DHAVE_NCCL
else
  $(info nccl.h not found: building without the NCCL baseline. Correctness only.)
endif

include ../../common/common.mk

NVCCFLAGS += $(EXTRA)
