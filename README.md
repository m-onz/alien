# alien 👽

![alien](/alien.png)

## About

An audio visual toolkit for live coding in Pure Data and GEM. The primary core of alien is the alien pure data external which is a live coding Pure Data external that produces sequences for `else/sequencer`. Based on my earlier algorithmic experiments (for example mixtape and check my github) the patterns accepted by else/sequencer: - - 2 3 - - 4 (hyphens and numbers) that are sequenced via a metro are sufficient for any complex piece of audio or visual algorithmic output. These patterns can be fed to to any parameter, audio or visual object or sub patch and it serves as the primary algorithmic mechanism within this system.

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
// - - 60 - - 64 - 67

; Random walk quantized to C major scale
(quantize (drunk 8 3 60) (seq 60 62 64 65 67 69 71 72))
// 60 62 64 62 65 67 69 67

; Arpeggio with delay
(delay (arp (chord 60 0) 0 8) 4)
// - - - - 60 64 67 60 64 67 60 64
```

## Contributing

This has only been tested on a recent mac, if you have any trouble compiling this on windows and different linux OS please submit a bug report or patch.

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

The alien pattern DSL is a lisp like syntax and alien is named after the Lisp alien mascot from [lisp](https://lispers.org/)
