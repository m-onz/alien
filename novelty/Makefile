# Makefile for ns_* — novelty-search externals for Pure Data
# Mirrors the conventions of ../Makefile.

PREFIX ?=

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    EXT = pd_darwin
    ARCH_FLAGS = -arch x86_64 -arch arm64
    LDFLAGS = -bundle -undefined dynamic_lookup
    DEFAULT_PD_DIR = ~/Documents/Pd/externals
endif
ifeq ($(UNAME),Linux)
    EXT = pd_linux
    ARCH_FLAGS = -fPIC
    LDFLAGS = -shared
    DEFAULT_PD_DIR = ~/.local/lib/pd/extra
endif
ifeq ($(OS),Windows_NT)
    EXT = dll
    ARCH_FLAGS =
    LDFLAGS = -shared
    DEFAULT_PD_DIR = $(APPDATA)/Pd
endif

ifeq ($(PREFIX),)
    PD_EXTERNALS_DIR = $(DEFAULT_PD_DIR)
else
    PD_EXTERNALS_DIR = $(PREFIX)
endif

CC = gcc
PD_INCLUDES = -I/usr/local/include -I/Applications/Pd-0.56-1.app/Contents/Resources/src
WARNINGS = -Wall -W -Wno-unused -Wno-parentheses -Wno-switch
OPTFLAGS = -O3 -funroll-loops -fomit-frame-pointer
PD_CFLAGS = $(ARCH_FLAGS) $(PD_INCLUDES) $(WARNINGS) $(OPTFLAGS) -DPD
CLI_CFLAGS = -Wall -Wextra -std=c99 -O2

EXTERNALS = ns_archive.$(EXT) ns_mutate.$(EXT) ns_log.$(EXT) ns_grid_stats.$(EXT) ns_spigot.$(EXT) ns_seq_features.$(EXT) ns_seq_propose.$(EXT) ns_seq_info.$(EXT) ns_quality.$(EXT) ns_corpus.$(EXT) ns_ast_features.$(EXT)

.PHONY: all build clean install test help

# Default target: build AND install. The stale-binary problem (Pd loading old
# copies from ~/Documents/Pd/externals/novelty/ instead of the freshly-built
# ones here) was a recurring footgun, so `make` deploys by default. Use
# `make build` if you only want to compile without copying.
all: build install

build: $(EXTERNALS) ns_parser ns_system_test validate_seeds

ns_archive.$(EXT): ns_archive.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_archive.c $(LDFLAGS) -lm

ns_mutate.$(EXT): ns_mutate.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_mutate.c $(LDFLAGS) -lm

ns_log.$(EXT): ns_log.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_log.c $(LDFLAGS) -lm

ns_grid_stats.$(EXT): ns_grid_stats.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_grid_stats.c $(LDFLAGS) -lm

ns_spigot.$(EXT): ns_spigot.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_spigot.c $(LDFLAGS) -lm

ns_seq_features.$(EXT): ns_seq_features.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_seq_features.c $(LDFLAGS) -lm

ns_seq_propose.$(EXT): ns_seq_propose.c ns_core.h ns_alien_ast.h ../alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_seq_propose.c $(LDFLAGS) -lm

ns_seq_info.$(EXT): ns_seq_info.c ns_core.h ns_alien_ast.h ../alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_seq_info.c $(LDFLAGS) -lm

ns_quality.$(EXT): ns_quality.c ns_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_quality.c $(LDFLAGS) -lm

ns_corpus.$(EXT): ns_corpus.c ns_core.h ../alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_corpus.c $(LDFLAGS) -lm

ns_ast_features.$(EXT): ns_ast_features.c ns_core.h ns_alien_ast.h ../alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ ns_ast_features.c $(LDFLAGS) -lm

ns_parser: ns_parser.c ns_core.h ns_alien_ast.h ../alien_core.h
	$(CC) $(CLI_CFLAGS) -o $@ ns_parser.c -lm

ns_system_test: ns_system_test.c ns_core.h ns_alien_ast.h ../alien_core.h
	$(CC) $(CLI_CFLAGS) -o $@ ns_system_test.c -lm

validate_seeds: validate_seeds.c ../alien_core.h
	$(CC) $(CLI_CFLAGS) -o $@ validate_seeds.c -lm

test: ns_parser
	./ns_parser --test

install: $(EXTERNALS)
	mkdir -p $(PD_EXTERNALS_DIR)/novelty
	cp $(EXTERNALS) $(PD_EXTERNALS_DIR)/novelty/
	@if [ -f ns-help.pd ]; then cp ns-help.pd $(PD_EXTERNALS_DIR)/novelty/; fi
	@echo "Installed to $(PD_EXTERNALS_DIR)/novelty"

clean:
	rm -f $(EXTERNALS) ns_parser ns_system_test validate_seeds *.o

help:
	@echo "ns_* — novelty-search externals for Pure Data"
	@echo ""
	@echo "Targets:"
	@echo "  make                 build everything + install (default)"
	@echo "  make build           build only — compile, don't install"
	@echo "  make install         copy built externals to $(PD_EXTERNALS_DIR)/novelty"
	@echo "  make test            run unit tests"
	@echo "  make clean           remove build artifacts"
	@echo ""
	@echo "Externals:"
	@echo "  ns_archive       vector archive + kNN novelty scoring"
	@echo "  ns_mutate        gaussian / bit-flip list perturbation"
	@echo "  ns_log           CSV / JSONL append writer"
	@echo "  ns_grid_stats    2D grid → 6-dim BC"
	@echo "  ns_spigot        threshold-gated list passer"
	@echo "  ns_seq_features  sequence (notes + rests) → 27-dim BC"
	@echo "  ns_seq_propose   AST-level alien DSL mutation/crossover/random"
	@echo "  ns_seq_info      alien DSL expr → (size, depth)"
