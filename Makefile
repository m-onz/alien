# Makefile for alien - Lisp pattern language for Pure Data
# Builds both Pure Data external and standalone CLI tool

# Allow user to override installation directory
# Usage: make install PREFIX=/custom/path
PREFIX ?=

# Platform detection
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

# Use PREFIX if set, otherwise use default
ifeq ($(PREFIX),)
    PD_EXTERNALS_DIR = $(DEFAULT_PD_DIR)
else
    PD_EXTERNALS_DIR = $(PREFIX)
endif

# Compiler and flags
CC = gcc
PD_INCLUDES = -I/usr/local/include -I/Applications/Pd-0.56-1.app/Contents/Resources/src
WARNINGS = -Wall -W -Wno-unused -Wno-parentheses -Wno-switch
OPTFLAGS = -O3 -funroll-loops -fomit-frame-pointer
PD_CFLAGS = $(ARCH_FLAGS) $(PD_INCLUDES) $(WARNINGS) $(OPTFLAGS) -DPD
CLI_CFLAGS = -Wall -Wextra -std=c99 -O2

# Build targets
.PHONY: all clean install test test-evo help

all: alien.$(EXT) alien_router.$(EXT) alien_router2.$(EXT) alien_scale.$(EXT) alien_groove.$(EXT) alien_cluster.$(EXT) alien_buffer.$(EXT) alien_parser

# Pure Data externals
alien.$(EXT): alien.c alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ alien.c $(LDFLAGS) -lm

alien_router.$(EXT): alien_router.c
	$(CC) $(PD_CFLAGS) -o $@ alien_router.c $(LDFLAGS)

alien_router2.$(EXT): alien_router2.c
	$(CC) $(PD_CFLAGS) -o $@ alien_router2.c $(LDFLAGS)

alien_scale.$(EXT): alien_scale.c alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ alien_scale.c $(LDFLAGS)

alien_groove.$(EXT): alien_groove.c
	$(CC) $(PD_CFLAGS) -o $@ alien_groove.c $(LDFLAGS)

alien_cluster.$(EXT): alien_cluster.c
	$(CC) $(PD_CFLAGS) -o $@ alien_cluster.c $(LDFLAGS) -lm

alien_buffer.$(EXT): alien_buffer.c
	$(CC) $(PD_CFLAGS) -o $@ alien_buffer.c $(LDFLAGS)

# Standalone CLI tools
alien_parser: alien_parser.c alien_core.h
	$(CC) $(CLI_CFLAGS) -o $@ alien_parser.c -lm

# Run tests
test: alien_parser
	./alien_parser --test

# Install Pure Data externals
install: alien.$(EXT) alien_router.$(EXT) alien_router2.$(EXT) alien_scale.$(EXT) alien_groove.$(EXT) alien_cluster.$(EXT) alien_buffer.$(EXT)
	mkdir -p $(PD_EXTERNALS_DIR)/alien
	cp alien.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp alien_router.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp alien_router2.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp alien_scale.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp alien_groove.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp alien_cluster.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp alien_buffer.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	@if [ -f examples/alien-help.pd ]; then \
		cp examples/alien-help.pd $(PD_EXTERNALS_DIR)/alien/; \
	elif [ -f alien-help.pd ]; then \
		cp alien-help.pd $(PD_EXTERNALS_DIR)/alien/; \
	else \
		echo "Warning: alien-help.pd not found, skipping"; \
	fi
	@if [ -f alien_router-help.pd ]; then \
		cp alien_router-help.pd $(PD_EXTERNALS_DIR)/alien/; \
	fi
	@echo "Installed to $(PD_EXTERNALS_DIR)/alien"

# Clean build artifacts
clean:
	rm -f alien.$(EXT) alien_router.$(EXT) alien_router2.$(EXT) alien_scale.$(EXT) alien_groove.$(EXT) alien_cluster.$(EXT) alien_buffer.$(EXT) alien_parser *.o

# Help
help:
	@echo "alien - Lisp pattern language for Pure Data"
	@echo ""
	@echo "Targets:"
	@echo "  make                 - Build all PD externals and CLI tools"
	@echo "  make alien.$(EXT)     - Build main PD external only"
	@echo "  make alien_parser    - Build pattern CLI tool"
	@echo "  make test            - Run pattern test suite"
	@echo "  make install         - Install PD externals to $(PD_EXTERNALS_DIR)"
	@echo "  make clean           - Remove build artifacts"
	@echo ""
	@echo "Installation Options:"
	@echo "  PREFIX=/path      - Custom installation directory"
	@echo "  Example: make install PREFIX=~/.pd-externals"
	@echo "  Default: $(DEFAULT_PD_DIR)"
	@echo ""
	@echo "CLI Usage (alien_parser):"
	@echo "  ./alien_parser '(euclid 5 8)'"
	@echo "  echo '(seq 1 2 3)' | ./alien_parser"
	@echo "  ./alien_parser --test"
