# Shared build rules for the prerequisite exercises.
#
# No CUDA. No GPU. A plain C++ compiler, which your Mac already has.
# Each exercise Makefile is two lines:
#
#   EXERCISE := p1-pointers
#   include ../../common/common.mk
#
# Targets:
#   make          build
#   make run      build and run the checks
#   make debug    build with sanitizers and no optimisation, then run
#   make clean

CXX      ?= c++
COMMON   := $(dir $(lastword $(MAKEFILE_LIST)))
BIN      := exercise

# -Wall -Wextra because at this stage the compiler's warnings are the most
# patient teacher you have. Do not silence them; read them.
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra -I$(COMMON)

# The debug build turns on the address and undefined-behaviour sanitizers.
# They catch exactly the mistakes this part of the course is about: reading off
# the end of an array, using memory you freed, signed overflow. Apple clang and
# gcc both support these.
DEBUGFLAGS := -std=c++17 -O0 -g -Wall -Wextra -fsanitize=address,undefined \
              -fno-omit-frame-pointer -I$(COMMON)

SRCS := main.cpp
DEPS := $(wildcard *.cpp *.h) $(COMMON)check.h

.PHONY: all run debug clean

all: $(BIN)

$(BIN): $(DEPS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@

run: $(BIN)
	./$(BIN) $(ARGS)

debug: $(DEPS)
	$(CXX) $(DEBUGFLAGS) $(SRCS) -o $(BIN)-debug
	./$(BIN)-debug $(ARGS)

clean:
	rm -f $(BIN) $(BIN)-debug
	rm -rf *.dSYM
