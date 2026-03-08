# alien

![alien](/alien.png)

A **Lisp-like pattern language** for algorithmic music in [Pure Data](https://puredata.info). Write expressive patterns as S-expressions, evaluate them to lists, and feed them to sequencers.

```
(euclid (seq 60 64 67) 8)  →  60 - - 64 - - 67 -
```

## Quick Start

### 1. Install

```bash
cd alien
make
make install
```

### 2. Configure Pd

Add the alien folder to your path: **File → Preferences → Path**

### 3. Make Sound

```
[alien kick]              [alien hihat]
    |                         |
[else/sequencer]          [else/sequencer]
    |                         |
[kick-synth~]             [hihat-synth~]
```

Send patterns via `[s kick]` and `[s hihat]`:

```
;kick (euclid 4 16)
;hihat (euclid 7 16)
```

## How It Works

`[alien]` has two modes:

### Generator Mode — `[alien]`

Expression in the inlet, list out. Stateless, immediate.

```
[alien]  ←  (seq 60 64 67 72)
   |
60 64 67 72
```

### Named Mode — `[alien kick]`

Receives patterns via `[s kick]`, evaluates the expression, and outputs the list. Multiple named aliens share an **entangled sync clock** — pattern changes arriving within 5ms are batched so all instances output simultaneously.

```
;kick (euclid (seq 36 - 36 -) 16)
;snare (euclid 3 16 4)
;hihat (euclid 7 16)
```

All three evaluate and output their new patterns at the same instant. Use `[t b b b]` to trigger the pattern sends together for guaranteed sync.

### Broadcast

Send a pattern to **all** named aliens at once:

```
;all (seq -)
```

This clears every named alien. Any expression works:

```
;all (euclid 4 16)
```

### Messages

| Message | Effect |
|---------|--------|
| `bang` | Re-output the current pattern |
| `unsync` | Leave the entangled group |
| `sync` | Re-join the entangled group |
| `reset` | Clear the stored pattern |

### Creation Arguments

| Object | Mode |
|--------|------|
| `[alien]` | Generator — expression in, list out |
| `[alien kick]` | Named — entangled, receives via `[s kick]` |
| `[alien kick -unsync]` | Named — independent (no sync batching) |

## The Pattern Language

Patterns are S-expressions that evaluate to sequences of integers. They nest arbitrarily:

```
(interleave (euclid 3 8) (seq 60 64 67 72 60 64 67 72))
```

The hyphen `-` represents a rest (also `.` and `_`). Note names like `C4`, `D#4`, `Bb3` parse to MIDI values. `x` is a hit (value 1).

### Core Operators

| Category | Operators |
|----------|-----------|
| **Sequence** | `seq`, `rep`, `range`, `ramp` |
| **Rhythm** | `euclid`, `bjork`, `subdiv` |
| **Transform** | `reverse`, `rotate`, `palindrome`, `mirror`, `interleave`, `shuffle` |
| **Select** | `take`, `drop`, `slice`, `every`, `filter` |
| **Math** | `add`, `mul`, `mod`, `scale`, `clamp`, `wrap`, `fold` |
| **Random** | `choose`, `rand`, `prob`, `maybe`, `drunk`, `degrade` |
| **Musical** | `chord`, `arp`, `transpose`, `quantize` |
| **Structure** | `cycle`, `grow`, `delay`, `gate` |

**See [REFERENCE.md](REFERENCE.md) for complete documentation with examples.**

### Examples

```lisp
; Euclidean rhythm with melody
(euclid (seq 60 64 67 72) 16)

; Chord progression
(seq (chord 60 0) (chord 65 0) (chord 67 0))

; Probabilistic pattern
(maybe (euclid 5 8) (euclid 3 8) 50)

; Generative melody
(quantize (drunk 16 3 60) (seq 48 50 52 53 55 57 59))

; Polyrhythm layers
;kick  (euclid 4 16)
;snare (euclid 3 16 4)
;hihat (euclid 7 16)
```

## Workflow

```
[s kick]  [s snare]  [s hihat]
    ↓          ↓          ↓
[alien kick] [alien snare] [alien hihat]    ← entangled: sync together
    ↓          ↓          ↓
[else/sequencer] ...                        ← stepping handled externally
    ↓
[synth~]
```

Pattern changes sent within 5ms are batched. All entangled aliens evaluate and output their lists at the same time, so sequencers receive new patterns in sync.

## CLI Tool

Test patterns without Pd:

```bash
./alien_parser '(euclid 5 8)'
./alien_parser '(interleave (seq 1 2 3) (seq 4 5 6))'
./alien_parser --test
```

## Installation

```bash
make                                 # build external + CLI
make install                         # install to ~/Documents/Pd/externals
make install PREFIX=~/.pd-externals  # custom path
make test                            # run test suite
make clean                           # remove build artifacts
```

## Theme

The `theme/` folder contains a dark canvas theme. Copy to your Pd externals folder.

## Credits

Named after the [Lisp alien](https://lispers.org/). Inspired by TidalCycles, SuperCollider, and the live coding community.

## License

MIT
