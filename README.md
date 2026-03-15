# alien

![alien](/alien.png)

A **Lisp-like pattern language** for algorithmic music in [Pure Data](https://puredata.info). Write expressive patterns as S-expressions, evaluate them to lists, and feed them to sequencers.

```lisp
(euclid (seq 60 64 67) 8)  →  60 - - 64 - - 67 -
```

See [this repo for the complete toolkit](https://github.com/m-onz/alien-av-toolkit)

## Installation

```bash
cd alien
make                                 # build external + CLI
make install                         # install to ~/Documents/Pd/externals
make install PREFIX=~/.pd-externals  # custom path
```

## Usage

### Generator Mode

```
[alien]
```

Send a pattern, get a list:

```
(seq 1 2 3)  →  1 2 3
```

### Sequencer Mode

```
[alien kick snare hihat]
```

- One outlet per voice, plus loop bang
- Send patterns via `[; kick (euclid 4 16)]`
- Bang to advance step

---

## Pattern Language Reference

All operators use prefix notation: `(operator arg1 arg2 ...)`. Patterns nest freely — any argument can be another expression. The `-` character represents a rest (silence).

---

### Basics

| Syntax | Description |
|--------|-------------|
| `60` | A number (e.g. MIDI note) |
| `-` | A rest (silence) |

```lisp
60                              → 60
-                               → -
```

---

### seq — Sequence

Build a sequence from values or sub-expressions. This is how you group things together.

```lisp
(seq 1 2 3)                     → 1 2 3
(seq 60 - 64 - 67)              → 60 - 64 - 67
(seq)                            → (empty)
(seq (seq 1 2) (seq 3 4))       → 1 2 3 4     ; nested seqs flatten
```

---

### rep — Repeat

Repeat a value or sequence `n` times.

```lisp
(rep value count)

(rep 1 4)                        → 1 1 1 1
(rep (seq 1 2) 3)                → 1 2 1 2 1 2
(rep - 4)                        → - - - -
(rep (seq 60 64 67) 2)           → 60 64 67 60 64 67
```

---

### Arithmetic

All arithmetic preserves rests — rests pass through unchanged.

#### add — Add

```lisp
(add sequence n)

(add (seq 60 62 64) 12)          → 72 74 76     ; transpose up octave
(add (seq 1 - 3) 10)             → 11 - 13      ; rests preserved
```

#### sub — Subtract

```lisp
(sub sequence n)

(sub (seq 72 76 79) 12)          → 60 64 67     ; transpose down octave
(sub (seq 0) 1)                  → -1            ; negative values (not rests)
(sub (seq 72 - 64) 12)           → 60 - 52
```

#### mul — Multiply

```lisp
(mul sequence n)

(mul (seq 1 2 3) 2)              → 2 4 6
(mul (seq 1 2 3) 0)              → 0 0 0
(mul (seq 1 - 3) 5)              → 5 - 15
```

#### mod — Modulo

```lisp
(mod sequence n)

(mod (seq 8 9 10) 7)             → 1 2 3
(mod (seq 10 11 12) 12)          → 10 11 0
(mod (seq 1 - 5) 3)              → 1 - 2
```

---

### scale — Map Range

Linearly map values from one range to another.

```lisp
(scale sequence from_min from_max to_min to_max)

(scale (seq 0 64 127) 0 127 0 100)    → 0 50 100    ; MIDI to percent
(scale (seq 0 5 10) 0 10 60 72)       → 60 66 72     ; map to MIDI
(scale (seq - 5 -) 0 10 0 100)        → - 50 -       ; rests preserved
```

---

### clamp — Clamp Range

Constrain values to a min/max range.

```lisp
(clamp sequence min max)

(clamp (seq 1 5 10) 3 8)          → 3 5 8
(clamp (seq 0 50 127) 20 100)     → 20 50 100
(clamp (seq 1 - 10) 3 8)          → 3 - 8
```

---

### wrap — Modular Wrap

Values that exceed the range wrap around (modulo-style).

```lisp
(wrap sequence min max)

(wrap (seq 0 5 10 15 20) 0 12)   → 0 5 10 3 8
(wrap (seq 13 25 37) 0 12)       → 1 1 1
(wrap (seq 60 72 84) 60 72)      → 60 60 60
(wrap (seq 1 - 5) 0 10)          → 1 - 5
```

---

### fold — Fold / Reflect

Values that exceed the range bounce (reflect) back.

```lisp
(fold sequence min max)

(fold (seq 0 5 10 15 20) 0 10)   → 0 5 10 5 0
(fold (seq 50 60 70 80) 55 75)   → 60 60 70 70
(fold (seq 1 - 8) 0 10)          → 1 - 8
```

---

### Rhythm

#### euclid — Euclidean Rhythm

Distribute hits evenly across steps using the Euclidean algorithm.

```lisp
(euclid hits steps)
(euclid hits steps rotation)
(euclid hits steps rotation hit_value)
(euclid pattern steps)
(euclid pattern steps rotation)

; Basic: distribute hits, outputs 1 for each hit
(euclid 3 8)                           → - - 1 - - 1 - 1
(euclid 4 4)                           → 1 1 1 1
(euclid 1 4)                           → - - - 1
(euclid 5 8)                           → - 1 - 1 1 - 1 1
(euclid 0 8)                           → - - - - - - - -

; Rotation: shift the pattern
(euclid 3 8 2)                         → 1 - - 1 - 1 - -

; Hit value: output a specific number instead of 1
(euclid 3 8 0 36)                      → - - 36 - - 36 - 36
(euclid 4 4 0 60)                      → 60 60 60 60

; Sequence as hits: cycle values across hit positions
(euclid (seq 60 64 67) 8)              → - - 60 - - 64 - 67
(euclid (seq 36 38 42) 8)              → - - 36 - - 38 - 42
```

#### subdiv — Subdivide

Repeat each element `n` times (rhythmic subdivision).

```lisp
(subdiv sequence n)

(subdiv (seq 1 2) 2)              → 1 1 2 2
(subdiv (seq 1 2 3) 3)            → 1 1 1 2 2 2 3 3 3
(subdiv (seq 60 - 64) 2)          → 60 60 - - 64 64
(subdiv (seq 1 2 3) 1)            → 1 2 3       ; identity
```

---

### List Manipulation

#### reverse

```lisp
(reverse sequence)

(reverse (seq 1 2 3))             → 3 2 1
(reverse (seq 1 - 3))             → 3 - 1
```

#### rotate

Rotate elements to the right by `n` positions.

```lisp
(rotate sequence n)

(rotate (seq 1 2 3 4) 1)          → 4 1 2 3
(rotate (seq 1 2 3 4) 2)          → 3 4 1 2
(rotate (seq 1 2 3 4) 0)          → 1 2 3 4     ; identity
```

#### interleave

Weave two sequences together, alternating elements.

```lisp
(interleave sequence_a sequence_b)

(interleave (seq 1 2 3) (seq - - -))   → 1 - 2 - 3 -
(interleave (seq 1) (seq 2))            → 1 2
(interleave (seq 1 2) (seq 10 20 30))   → 1 10 2 20 30
```

#### shuffle

Randomly reorder a sequence. Non-deterministic — different result each time.

```lisp
(shuffle sequence)

(shuffle (seq 1 2 3 4))           → (random order)
```

#### mirror

Create a palindrome — the sequence followed by itself reversed (without repeating the last element).

```lisp
(mirror sequence)

(mirror (seq 1 2 3))              → 1 2 3 2 1
(mirror (seq 1 2))                → 1 2 1
(mirror (seq 60 64 67 72))        → 60 64 67 72 67 64 60
(mirror (seq 1 - 3))              → 1 - 3 - 1
```

---

### Selection / Filtering

#### take — First N

```lisp
(take sequence n)

(take (seq 1 2 3 4 5) 3)          → 1 2 3
(take (seq 1 2 3) 5)              → 1 2 3       ; clamps to length
(take (seq 1 2 3) 0)              → (empty)
```

#### drop — Remove First N

```lisp
(drop sequence n)

(drop (seq 1 2 3 4 5) 2)          → 3 4 5
(drop (seq 1 2 3) 10)             → (empty)
(drop (seq 1 2 3) 0)              → 1 2 3       ; identity
```

#### slice — Sub-range

Extract elements from index `start` (inclusive) to `end` (exclusive).

```lisp
(slice sequence start end)

(slice (seq 10 20 30 40 50) 1 3)  → 20 30
(slice (seq 10 20 30 40 50) 0 2)  → 10 20
(slice (seq 10 20 30 40 50) 3 5)  → 40 50
```

#### every — Every Nth

Take every `n`th element, starting from the first.

```lisp
(every sequence n)

(every (seq 1 2 3 4 5 6) 2)       → 1 3 5
(every (seq 1 2 3 4 5 6) 3)       → 1 4
(every (seq 1 2 3 4) 1)           → 1 2 3 4     ; identity
```

#### filter — Remove Rests

Strip all rests from a sequence, keeping only values.

```lisp
(filter sequence)

(filter (seq 1 - 2 - 3))          → 1 2 3
(filter (seq - - -))              → (empty)
(filter (euclid 3 8))             → 1 1 1
```

---

### Pattern Generation

#### range

Generate a sequence of integers from `start` to `end` (inclusive).

```lisp
(range start end)
(range start end step)

(range 1 5)                        → 1 2 3 4 5
(range 0 8 2)                      → 0 2 4 6 8
(range 0 10 3)                     → 0 3 6 9
(range 60 72)                      → 60 61 62 63 64 65 66 67 68 69 70 71 72
(range 0 0)                        → 0
```

#### ramp

Interpolate linearly between two values over `count` points.

```lisp
(ramp start end count)

(ramp 60 72 5)                     → 60 63 66 69 72
(ramp 0 10 4)                      → 0 3 7 10
(ramp 10 0 3)                      → 10 5 0       ; descending
(ramp 0 0 3)                       → 0 0 0         ; flat
(ramp 0 12 7)                      → 0 2 4 6 8 10 12
```

---

### Randomness

All random operators are non-deterministic — they produce different results on each evaluation.

#### choose — Pick One

Randomly select one of its arguments. Each argument can be a value or expression.

```lisp
(choose arg1 arg2 ...)

(choose 60 64 67)                  → 60 (or 64, or 67)
(choose (seq 1 2) (seq 3 4))      → 1 2 (or 3 4)
```

#### rand — Random Values

Generate `count` random integers. Defaults to MIDI range 0-127.

```lisp
(rand count)
(rand count min max)

(rand 4)                           → (4 random values, 0-127)
(rand 4 60 72)                     → (4 random values, 60-72)
```

#### prob — Probabilistic Gate

Each element has a `percent`% chance of passing through. Failures become rests.

```lisp
(prob sequence percent)

(prob (seq 1 2 3 4) 50)           → (each has 50% chance)
(prob (seq 1 2 3 4) 100)          → 1 2 3 4     ; always
(prob (seq 1 2 3 4) 0)            → - - - -     ; never
```

#### drunk — Random Walk

Generate a random walk: `steps` values starting at `start`, moving by at most `max_step` each time. Optionally bounded to a `min`-`max` range (reflects at boundaries).

```lisp
(drunk steps max_step start)
(drunk steps max_step start min max)

(drunk 8 2 60)                     → (8 values, random walk from 60, ±2 each step)
(drunk 8 3 60 48 72)               → (bounded between 48 and 72)
(drunk 16 1 64 60 68)              → (gentle walk in narrow range)
```

---

### Musical

#### quantize — Snap to Scale

Snap each value to the nearest pitch class in a scale. Octave-aware — works correctly across the full MIDI range.

```lisp
(quantize sequence scale_degrees)

; Common scales as pitch classes (0 = C, 2 = D, etc.)
(quantize (seq 61 63 66) (seq 0 2 4 5 7 9 11))  → 60 62 65   ; C major
(quantize (seq 60) (seq 0 4 7))                   → 60         ; C major triad
(quantize (seq 1 6 10) (seq 0 4 7 11))            → 0 7 11     ; Cmaj7
```

#### arp — Arpeggiate

Generate an arpeggio pattern from a set of notes. Direction: `0` = up, `1` = down, `2` = up-down.

```lisp
(arp notes direction length)

(arp (seq 60 64 67) 0 6)           → 60 64 67 60 64 67    ; up
(arp (seq 60 64 67) 1 6)           → 67 64 60 67 64 60    ; down
(arp (seq 60 64 67) 2 8)           → 60 64 67 64 60 64 67 64  ; up-down
(arp (seq 60) 2 5)                 → 60 60 60 60 60        ; single note
```

---

### Time / Structure

#### cycle — Loop to Length

Repeat a pattern to fill a target length.

```lisp
(cycle sequence length)

(cycle (seq 1 2 3) 8)             → 1 2 3 1 2 3 1 2
(cycle (seq 1 2 3) 3)             → 1 2 3       ; exact fit
(cycle (seq 1 2 3) 1)             → 1            ; truncate
(cycle (seq 60 - 64) 6)           → 60 - 64 60 - 64
```

#### grow — Gradual Reveal

Progressively reveal more of a pattern. Each iteration shows one more element, with rests filling the remaining slots.

```lisp
(grow sequence)

(grow (seq 1 2 3))                 → 1 - - 1 2 - 1 2 3
(grow (seq 1 2))                   → 1 - 1 2
(grow (seq 60 64 67 72))           → 60 - - - 60 64 - - 60 64 67 - 60 64 67 72
```

#### gate — Rhythmic Gate

Keep every `n`th element, replace the rest with rests.

```lisp
(gate sequence n)

(gate (seq 1 2 3 4 5 6) 2)        → 1 - 3 - 5 -
(gate (seq 1 2 3 4 5 6) 3)        → 1 - - 4 - -
(gate (seq 1 2 3) 1)              → 1 2 3       ; identity
```

#### speed — Stretch

Insert `n-1` rests after each element, stretching the pattern.

```lisp
(speed sequence n)

(speed (seq 1 2 3) 2)             → 1 - 2 - 3 -
(speed (seq 1 2 3) 4)             → 1 - - - 2 - - - 3 - - -
(speed (seq 1 2 3) 1)             → 1 2 3       ; identity
(speed (seq 60) 3)                → 60 - -
```

#### mask — Pattern Gating

Use a gate pattern to selectively pass through values from a source. Where the gate has a value, take the next value from the source (cycling). Where the gate has a rest, output a rest.

```lisp
(mask source gate)

(mask (seq 60 64 67) (euclid 3 8))       → - - 60 - - 64 - 67
(mask (seq 1 2 3) (seq 1 - 1 - 1))       → 1 - 2 - 3
(mask (seq 1 2) (seq 1 1 1 1))            → 1 2 1 2    ; source cycles
(mask (seq 36 38) (euclid 2 4))           → - 36 - 38
```

#### delay — Prepend Rests

Add `n` rests before the sequence.

```lisp
(delay sequence n)

(delay (seq 1 2 3) 2)             → - - 1 2 3
(delay (seq 60) 4)                → - - - - 60
(delay (seq 1 2 3) 0)             → 1 2 3       ; identity
```

---

## Composition

Patterns nest freely — any argument can be another expression:

```lisp
; Arpeggiated euclidean rhythm
(euclid (arp (seq 60 64 67) 0 4) 16)

; Polyrhythmic interleave
(interleave (euclid 3 8) (euclid 5 8))

; Random transposition
(add (shuffle (seq 60 64 67 71)) (choose 0 12))

; Euclidean kick pattern with MIDI note
(euclid 4 16 0 36)

; Euclidean hi-hats with different sounds
(euclid (seq 42 44 46) 16)

; Distribute MIDI notes over a euclidean rhythm
(mask (seq 60 64 67 72) (euclid 4 16))

; Filtered random walk snapped to scale
(quantize (drunk 16 3 60 48 72) (seq 0 2 4 5 7 9 11))

; Gradually reveal a chord
(grow (seq 60 64 67 72))

; Reversed ramp as velocity curve
(reverse (ramp 40 127 16))

; Take a slice of a cycle
(slice (cycle (seq 60 64 67 72) 32) 8 16)

; Gate a range to create dotted rhythm
(gate (range 60 75) 3)

; Fold a drunk walk into a narrow range
(fold (drunk 16 5 60) 55 70)

; Speed up a euclidean pattern
(speed (euclid 3 8) 2)

; Mirror an arpeggio
(mirror (arp (seq 60 64 67 72) 0 4))
```

---

## CLI Tool

Test patterns without Pd:

```bash
./alien_parser '(euclid 5 8)'
./alien_parser '(seq 60 64 67)'
./alien_parser '(euclid 3 8 0 36)'
./alien_parser '(drunk 16 3 60)'
./alien_parser --test               # run test suite (223 tests)
```

---

## Examples

### Four-on-the-floor

```
[; kick (euclid 4 16 0 36)]
[; snare (euclid 2 16 4 38)]
[; hihat (rep 42 16)]
```

### Generative melody

```
[; lead (quantize (drunk 16 3 60 48 72) (seq 0 2 4 5 7 9 11))]
```

### Polyrhythm

```
[; a (euclid 3 8)]
[; b (euclid 5 8)]
[; c (euclid 7 8)]
```

### Build-up

```
[; perc (grow (seq 36 38 42 46))]
```

---

## Operator Quick Reference

| Operator | Args | Description |
|----------|------|-------------|
| `seq` | `val ...` | Sequence of values |
| `rep` | `val n` | Repeat `n` times |
| `add` | `seq n` | Add `n` to each value |
| `sub` | `seq n` | Subtract `n` from each value |
| `mul` | `seq n` | Multiply each value by `n` |
| `mod` | `seq n` | Modulo each value by `n` |
| `scale` | `seq fmin fmax tmin tmax` | Map from one range to another |
| `clamp` | `seq min max` | Constrain to range |
| `wrap` | `seq min max` | Modular wrap to range |
| `fold` | `seq min max` | Reflect/fold at range boundaries |
| `euclid` | `hits steps [rot [val]]` | Euclidean rhythm |
| `subdiv` | `seq n` | Subdivide each element `n` times |
| `reverse` | `seq` | Reverse order |
| `rotate` | `seq n` | Rotate right by `n` |
| `interleave` | `a b` | Alternate elements from two sequences |
| `shuffle` | `seq` | Random order |
| `mirror` | `seq` | Palindrome |
| `take` | `seq n` | First `n` elements |
| `drop` | `seq n` | Remove first `n` elements |
| `slice` | `seq start end` | Sub-range (start inclusive, end exclusive) |
| `every` | `seq n` | Every `n`th element |
| `filter` | `seq` | Remove rests |
| `range` | `start end [step]` | Integer range |
| `ramp` | `start end count` | Linear interpolation |
| `choose` | `arg ...` | Pick one argument at random |
| `rand` | `count [min max]` | Random values (default 0-127) |
| `prob` | `seq percent` | Probabilistic gate (0-100) |
| `drunk` | `steps max start [min max]` | Random walk |
| `quantize` | `seq scale` | Snap to nearest scale degree |
| `arp` | `seq dir len` | Arpeggiate (0=up, 1=down, 2=up-down) |
| `cycle` | `seq len` | Loop pattern to length |
| `grow` | `seq` | Gradually reveal pattern |
| `gate` | `seq n` | Keep every `n`th, rest others |
| `speed` | `seq n` | Stretch with rests |
| `mask` | `source gate` | Gate-controlled value selection |
| `delay` | `seq n` | Prepend `n` rests |

---

## Theme

The `theme/` folder contains a dark canvas theme.

## Credits

Named after the [Lisp alien](https://lispers.org/)

## License

MIT
