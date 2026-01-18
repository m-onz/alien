# alien 👽

![alien](/alien.png)

## About

An audio visual toolkit for live coding in Pure Data and GEM. The primary core of alien is the alien pure data external which produces sequences for `else/sequencer`. Based on my earlier algorithmic experiments (for example mixtape and check my github) the patterns accepted by else/sequencer: - - 2 3 - - 4 (hyphens and numbers) that are sequenced via a metro are sufficient for any complex piece of audio or visual algorithmic output. These patterns can be fed to to any parameter, audio or visual object or sub patch and it serves as the primary algorithmic mechanism within this system.

## Audio visuals

This is an audio visual toolkit that supports audio visual live coding but also parameterized algorithmic performance. Whereby the patterns have been specified ahead of time and mapped as parameters for control via a MIDI instrument or keyboard. I personally think both approaches have their merits and they are not mutually exclusive.

The issue with audio visual live coding is the ranges of numbers useful in music are not the same as in visuals. A co-ordinate system and movement of objects in 2D or 3D space requires numbers with a larger range than MIDI or rhythmic patterns. It is necessary to do mapping between these two spaces ahead of time or to future proof your abstractions with flexible internal mappings.

## Technical notes 

You will probably need to increase the delay in microseconds to 256 or higher and the block size to 1024 in the audio preferences to stop the visuals from glitching the audio in some cases (this may work better using an external sound card). It's possible to crash Pd if you try to delete a GEM object without a GEM window on a mac.

## Objects

* alien
* alien_router
* alien_monosynth~
* alien_monosynth2~
* video
* playdir~

## Features

* **Pattern Language External** - A Pure Data external (`[alien]`) implementing a Lisp-like pattern language with 39+ operators
* **Standalone CLI Tool** - `alien_parser` for testing patterns outside of Pure Data
* **Example Patches** - Starter patches showing audio visual setups
* **Comprehensive Documentation** - Full operator reference with examples

## Installation

### Requirements

* [Pure Data](http://msp.ucsd.edu/software.html) (tested with Pd-0.56-1)
* [pd-else](https://github.com/porres/pd-else) - Via deken - or install from source if you see any broken externals
* freeverb~ - Install via deken (pure data package manager)
* C compiler (gcc, clang) for building from source

### Build from Source

```bash
# Build everything (PD external + CLI tool)
make

# Run tests
make test

# Install Pure Data external (default location)
make install

# Install to custom directory
make install PREFIX=/path/to/pd-externals
```

**Default Installation Paths:**
- **macOS**: `~/Documents/Pd/externals/alien`
- **Linux**: `~/.local/lib/pd/extra/alien`
- **Windows**: `%APPDATA%/Pd/alien`

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

The alien pattern DSL is a lisp like syntax and alien is named after the [lisp alien mascot](https://lispers.org/)
