# Alien Pattern Language Reference

Complete reference for the `[alien]` pattern language — a Lisp-like expression system for generating sequences in Pure Data.

## Quick Start

```
(seq 1 2 3)           → 1 2 3
(euclid 3 8)          → 1 - - 1 - - 1 -
(chord 60 0)          → 60 64 67
(reverse (range 1 5)) → 5 4 3 2 1
```

Patterns are S-expressions. They nest arbitrarily:
```
(interleave (euclid 3 8) (seq 60 64 67 72 60 64 67 72))
```

The hyphen `-` represents a rest (outputs as `-1` internally, `-` symbol in Pd).

---

## Operators by Category

### Sequences

| Operator | Signature | Description |
|----------|-----------|-------------|
| `seq` | `(seq a b c ...)` | Concatenate values into a sequence |
| `rep` | `(rep pattern n)` | Repeat pattern n times |
| `range` | `(range start end [step])` | Generate integer range |
| `ramp` | `(ramp start end steps)` | Linear interpolation over n steps |

**Examples:**
```
(seq 1 2 3)         → 1 2 3
(seq 1 (seq 2 3) 4) → 1 2 3 4
(rep (seq 1 2) 3)   → 1 2 1 2 1 2
(range 0 4)         → 0 1 2 3 4
(range 10 0 -2)     → 10 8 6 4 2 0
(ramp 0 100 5)      → 0 25 50 75 100
```

---

### Rhythm

| Operator | Signature | Description |
|----------|-----------|-------------|
| `euclid` | `(euclid hits steps [rotation])` | Euclidean rhythm pattern |
| `euclid` | `(euclid pattern steps [rotation])` | Distribute pattern across Euclidean hits |
| `bjork` | `(bjork hits steps)` | Bjorklund algorithm (alternative distribution) |
| `subdiv` | `(subdiv pattern n)` | Subdivide each step n times |

**Examples:**
```
(euclid 3 8)              → 1 - - 1 - - 1 -
(euclid 5 8)              → 1 - 1 - 1 - 1 1
(euclid 3 8 2)            → - 1 - - 1 - - 1  (rotated)
(euclid (seq 60 64 67) 8) → 60 - - 64 - - 67 -
(bjork 3 8)               → 1 - - 1 - - 1 -
(subdiv (seq 1 2) 3)      → 1 1 1 2 2 2
```

---

### List Manipulation

| Operator | Signature | Description |
|----------|-----------|-------------|
| `reverse` | `(reverse pattern)` | Reverse the sequence |
| `rotate` | `(rotate pattern n)` | Rotate right by n steps |
| `palindrome` | `(palindrome pattern)` | Append reverse (excluding last) |
| `mirror` | `(mirror pattern)` | Append full reverse |
| `interleave` | `(interleave a b)` | Interleave two sequences |
| `shuffle` | `(shuffle pattern)` | Randomly shuffle |

**Examples:**
```
(reverse (seq 1 2 3))       → 3 2 1
(rotate (seq 1 2 3 4) 1)    → 4 1 2 3
(rotate (seq 1 2 3 4) -1)   → 2 3 4 1
(palindrome (seq 1 2 3))    → 1 2 3 2 1
(mirror (seq 1 2 3))        → 1 2 3 3 2 1
(interleave (seq 1 2) (seq a b)) → 1 a 2 b
(shuffle (range 1 5))       → (random order)
```

---

### Selection & Filtering

| Operator | Signature | Description |
|----------|-----------|-------------|
| `take` | `(take pattern n)` | First n elements |
| `drop` | `(drop pattern n)` | Remove first n elements |
| `slice` | `(slice pattern start end)` | Extract subsequence |
| `every` | `(every pattern n)` | Every nth element |
| `filter` | `(filter pattern)` | Remove rests (hyphens) |

**Examples:**
```
(take (range 1 10) 3)       → 1 2 3
(drop (range 1 5) 2)        → 3 4 5
(slice (range 0 9) 2 5)     → 2 3 4
(every (range 1 8) 2)       → 1 3 5 7
(filter (seq 1 - 2 - 3))    → 1 2 3
```

---

### Arithmetic

| Operator | Signature | Description |
|----------|-----------|-------------|
| `add` | `(add pattern n)` | Add n to each value |
| `mul` | `(mul pattern n)` | Multiply each value by n |
| `mod` | `(mod pattern n)` | Modulo n on each value |
| `scale` | `(scale pattern from_min from_max to_min to_max)` | Map value range |
| `clamp` | `(clamp pattern min max)` | Constrain values to range |

**Examples:**
```
(add (seq 60 64 67) 12)           → 72 76 79
(mul (seq 1 2 3) 10)              → 10 20 30
(mod (range 0 7) 4)               → 0 1 2 3 0 1 2 3
(scale (range 0 4) 0 4 60 72)     → 60 63 66 69 72
(clamp (range 0 10) 3 7)          → 3 3 3 3 4 5 6 7 7 7 7
```

---

### Randomness

| Operator | Signature | Description |
|----------|-----------|-------------|
| `choose` | `(choose a b c ...)` | Pick one argument randomly |
| `rand` | `(rand count min max)` | Generate n random values |
| `prob` | `(prob pattern percent)` | Keep each value with probability |
| `maybe` | `(maybe a b percent)` | Choose a with percent probability, else b |
| `drunk` | `(drunk steps max_step start)` | Random walk |
| `degrade` | `(degrade pattern percent)` | Replace values with rests |

**Examples:**
```
(choose 60 64 67)           → 60 or 64 or 67
(rand 4 0 127)              → (4 random values 0-127)
(prob (seq 1 2 3 4) 50)     → (each kept with 50% chance)
(maybe (seq 1 2) (seq 3 4) 75) → seq 1 2 (75%) or seq 3 4 (25%)
(drunk 8 2 60)              → (8-step random walk from 60, ±2)
(degrade (range 1 8) 30)    → (30% of values become rests)
```

---

### Musical

| Operator | Signature | Description |
|----------|-----------|-------------|
| `transpose` | `(transpose pattern n)` | Add n semitones (alias for add) |
| `quantize` | `(quantize pattern scale)` | Snap to nearest scale tone |
| `chord` | `(chord root type)` | Generate chord |
| `arp` | `(arp pattern direction length)` | Arpeggiate |

**Chord types:**
- `0` = major (0 4 7)
- `1` = minor (0 3 7)
- `2` = diminished (0 3 6)
- `3` = augmented (0 4 8)
- `4` = major 7th (0 4 7 11)
- `5` = minor 7th (0 3 7 10)
- `6` = dominant 7th (0 4 7 10)

**Arp directions:**
- `0` = up
- `1` = down
- `2` = up-down (ping-pong)

**Examples:**
```
(chord 60 0)                    → 60 64 67 (C major)
(chord 60 1)                    → 60 63 67 (C minor)
(chord 60 4)                    → 60 64 67 71 (Cmaj7)
(transpose (chord 60 0) 5)      → 65 69 72 (F major)
(arp (chord 60 0) 0 8)          → 60 64 67 60 64 67 60 64
(arp (chord 60 0) 2 8)          → 60 64 67 64 60 64 67 64
(quantize (rand 8 0 24) (seq 0 2 4 5 7 9 11)) → (snapped to C major)
```

---

### Time & Structure

| Operator | Signature | Description |
|----------|-----------|-------------|
| `cycle` | `(cycle pattern length)` | Repeat/truncate to exact length |
| `grow` | `(grow pattern)` | Progressive reveal |
| `delay` | `(delay pattern n)` | Prepend n rests |
| `gate` | `(gate pattern n)` | Keep every nth, rest others |

**Examples:**
```
(cycle (seq 1 2 3) 8)       → 1 2 3 1 2 3 1 2
(grow (seq 1 2 3))          → 1 - - 1 2 - 1 2 3
(delay (seq 1 2 3) 2)       → - - 1 2 3
(gate (range 1 8) 2)        → 1 - 3 - 5 - 7 -
```

---

## Composition Patterns

### Basic melody with Euclidean rhythm
```
(euclid (seq 60 64 67 72) 8)
```

### Chord progression
```
(seq (chord 60 0) (chord 65 0) (chord 67 0) (chord 60 0))
```

### Probabilistic variation
```
(maybe (euclid 5 8) (euclid 3 8) 50)
```

### Layered polyrhythm
```
kick:  (euclid 4 16)
snare: (euclid 3 16 4)
hat:   (euclid 7 16)
```

### Generative melody
```
(quantize (drunk 16 3 60) (add (seq 0 2 4 5 7 9 11) 48))
```

---

## Workflow

```
;kick (euclid 4 16)         ← send pattern via message
    ↓
[alien kick]                ← evaluates, outputs list
    ↓
[else/sequencer]            ← steps through the list
    ↓
[synth~]
```

Multiple named aliens are **entangled** — pattern changes arriving within 5ms are batched so all instances output simultaneously. Use `[t b b b]` to trigger sends together.

Broadcast to all named aliens:
```
;all (seq -)                ← clear all
;all (euclid 4 16)          ← set all to same pattern
```

---

## Version

alien v0.3.1
