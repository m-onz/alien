# alien 👽

An audio visual toolkit for live coding in Pure Data and GEM.

![alien](/alien.png)

## About

alien is a live coding Pure Data external that produces sequences for `else/sequencer` and other PD objects.

Based around a Lisp-like syntax, the mascot of Lisp is the [lisp alien](https://lispers.org/), so I decided to call my AV framework: "alien".

There are very few live coding or algorithmic music or visual systems. Typically you will get a purpose-built system for audio or visuals. Very few libraries, frameworks or solutions address the challenging problem of making audio visual performances.

## Pure Data + GEM

Pure Data (GEM) is my weapon of choice and it has never let me down. I know you can get better support for shaders with Touch Designer, and Tidal Cycles and SuperCollider have more flexible and powerful audio engines. However, for an audio visual platform, Pure Data and GEM has many advantages over commercial or open source alternatives:

* Everything runs in the same context - no client/server OSC latency
* It's old, doesn't change, and can be extended
* It's easy to install on any device

## Features

* **Pattern Language External** - A Pure Data external (`[alien]`) implementing a Lisp-like pattern language with 39+ operators
* **Standalone CLI Tool** - `alien_parser` for testing patterns outside of Pure Data
* **Example Patches** - Starter patches showing audio visual setups (COMING SOON)
* **Comprehensive Documentation** - Full operator reference with examples

## Installation

### Requirements

* [Pure Data](http://msp.ucsd.edu/software.html) (tested with Pd-0.56-1)
* [pd-else](https://github.com/porres/pd-else) - Required external library
* C compiler (gcc, clang) for building from source

### Build from Source

```bash
# Build everything (PD external + CLI tool)
make

# Run tests
make test

# Install Pure Data external
make install
```

## Quick Start

### Pure Data

Create a new patch and use the `[alien]` object:

```
[symbol (euclid 5 8)(
|
[alien]
|
[print]
```

Output: `- 1 - 1 1 - 1 1`

See `alien-help.pd` for a comprehensive help patch.

### Command Line

You can test the alien parser in isolation using the standalone cli

```bash
# Generate a pattern
./alien_parser "(euclid 5 8)"
# Output: - 1 - 1 1 - 1 1

# Pipe expressions
echo "(interleave (euclid 3 8) (range 60 67))" | ./alien_parser

# Run tests
./alien_parser --test
```

## Pattern Language

The alien pattern language provides 39+ operators organized into categories:

- **Core** (2): `seq`, `rep`
- **Arithmetic** (5): `add`, `mul`, `mod`, `scale`, `clamp`
- **Rhythm** (3): `euclid`, `bjork`, `subdiv`
- **List Manipulation** (6): `reverse`, `rotate`, `palindrome`, `mirror`, `interleave`, `shuffle`
- **Selection** (5): `take`, `drop`, `every`, `slice`, `filter`
- **Randomness** (5): `choose`, `rand`, `prob`, `maybe`, `degrade`
- **Pattern Generation** (3): `range`, `ramp`, `drunk`
- **Logic** (2): `cycle`, `grow`
- **Musical** (4): `transpose`, `quantize`, `chord`, `arp`
- **Time/Phase** (2): `delay`, `gate`

### Examples

```lisp
; Euclidean rhythm with notes
(euclid (seq 60 64 67) 8)
→ - - 60 - - 64 - 67

; Random walk quantized to C major scale
(quantize (drunk 8 3 60) (seq 60 62 64 65 67 69 71 72))
→ 60 62 64 62 65 67 69 67

; Arpeggio with delay
(delay (arp (chord 60 0) 0 8) 4)
→ - - - - 60 64 67 60 64 67 60 64
```

See [`docs/OPERATORS.md`](docs/OPERATORS.md) for complete operator reference.

## Project Structure

```
alien/
├── alien.c              - Pure Data external (uses alien_core.h)
├── alien_parser.c       - Standalone CLI tool (uses alien_core.h)
├── alien_core.h         - Shared pattern language implementation
├── Makefile             - Build system
├── README.md            - This file
├── LICENSE              - MIT License
├── examples/
│   └── alien-help.pd    - Pure Data help patch
└── docs/
    └── OPERATORS.md     - Complete operator reference
```

## Project Aims

To create an audio visual framework for Pure Data and GEM with batteries included. Supporting algorithmic live coding for audio visuals, shaders and more.

The main challenge with getting started with Pure Data and GEM is battling with the different options, so alien will provide starter patches as a starting point for experimentation. These patches will be completely flexible to incorporate other externals/libraries and approaches.

If you don't want to spend your time inventing a live coding or algorithmic paradigm and just want to make cool audio visual performances, alien presents a viable approach that has been years in the making.

## Contributing

I really want people to try compiling this on windows and different linux OS and report bugs and errors or contribute patches.

To add new operators to the alien DSL:

1. Add `NODE_XXX` to `NodeType` enum in `alien_core.h`
2. Implement `eval_xxx()` function
3. Add to parser switch statement
4. Add test cases to `alien_parser.c`
5. Document in `docs/OPERATORS.md`

## License

MIT License - see LICENSE file for details.

## Credits

Pattern language design inspired by TidalCycles, SuperCollider, and live coding communities.

The Lisp alien mascot is from [lispers.org](https://lispers.org/)
