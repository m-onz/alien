# Makefile for alien - Lisp pattern language for Pure Data
# Builds both Pure Data external and standalone CLI tool

# Platform detection
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    EXT = pd_darwin
    ARCH_FLAGS = -arch x86_64 -arch arm64
    LDFLAGS = -bundle -undefined dynamic_lookup
    PD_EXTERNALS_DIR = ~/Documents/Pd/externals
endif
ifeq ($(UNAME),Linux)
    EXT = pd_linux
    ARCH_FLAGS = -fPIC
    LDFLAGS = -shared
    PD_EXTERNALS_DIR = ~/.local/lib/pd/extra
endif
ifeq ($(OS),Windows_NT)
    EXT = dll
    ARCH_FLAGS =
    LDFLAGS = -shared
    PD_EXTERNALS_DIR = $(APPDATA)/Pd
endif

# Compiler and flags
CC = gcc
PD_INCLUDES = -I/usr/local/include -I/Applications/Pd-0.56-1.app/Contents/Resources/src
WARNINGS = -Wall -W -Wno-unused -Wno-parentheses -Wno-switch
OPTFLAGS = -O3 -funroll-loops -fomit-frame-pointer
PD_CFLAGS = $(ARCH_FLAGS) $(PD_INCLUDES) $(WARNINGS) $(OPTFLAGS) -DPD
CLI_CFLAGS = -Wall -Wextra -std=c99 -O2

# Build targets
.PHONY: all clean install test help

all: alien.$(EXT) alien_parser

# Pure Data external
alien.$(EXT): alien.c alien_core.h
	$(CC) $(PD_CFLAGS) -o $@ alien.c $(LDFLAGS) -lm

# Standalone CLI tool
alien_parser: alien_parser.c alien_core.h
	$(CC) $(CLI_CFLAGS) -o $@ alien_parser.c -lm

# Run tests
test: alien_parser
	./alien_parser --test

# Install Pure Data external
install: alien.$(EXT)
	mkdir -p $(PD_EXTERNALS_DIR)/alien
	cp alien.$(EXT) $(PD_EXTERNALS_DIR)/alien/
	cp examples/alien-help.pd $(PD_EXTERNALS_DIR)/alien/
	@echo "Installed to $(PD_EXTERNALS_DIR)/alien"

# Clean build artifacts
clean:
	rm -f alien.$(EXT) alien_parser *.o

# Help
help:
	@echo "alien - Lisp pattern language for Pure Data"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build both PD external and CLI tool"
	@echo "  make alien.$(EXT)  - Build Pure Data external only"
	@echo "  make alien_parser - Build CLI tool only"
	@echo "  make test         - Run test suite"
	@echo "  make install      - Install PD external to $(PD_EXTERNALS_DIR)"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "CLI Usage:"
	@echo "  ./alien_parser '(euclid 5 8)'"
	@echo "  echo '(seq 1 2 3)' | ./alien_parser"
	@echo "  ./alien_parser --test"
