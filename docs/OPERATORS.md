# Alien - Lisp Pattern Language for Pure Data

A comprehensive pattern generator for Pure Data with 39+ operators for live coding, algorithmic composition, and sequence generation.

## Installation

### Pure Data External
```bash
make alien
# Install to Pd externals directory:
make install
```

### Standalone CLI Tool
```bash
make seqgen
./seqgen --test              # Run tests
./seqgen "(euclid 5 8)"      # Generate pattern
```

## Usage

### Pure Data
```
[symbol (seq 1 2 3)(
|
[alien]
|
[print]  -> outputs: 1 2 3
```

### Command Line
```bash
./seqgen "(euclid 5 8)"
# Output: - 1 - 1 1 - 1 1

echo "(interleave (euclid 3 8) (range 60 67))" | ./seqgen
```

## Operator Reference

### Core Operators (2)

**`seq`** - Flatten and concatenate sequences
```
(seq 1 2 3)                    → 1 2 3
(seq (seq 1 2) (seq 3 4))      → 1 2 3 4
(seq 1 2 (seq 3 4))            → 1 2 3 4
```

**`rep`** - Repeat pattern N times (last arg is count)
```
(rep 1 3)                      → 1 1 1
(rep (seq 1 2) 4)              → 1 2 1 2 1 2 1 2
(rep (seq 1 2) (seq 3 4) 2)    → 1 2 3 4 1 2 3 4
```

### Arithmetic Operators (5)

**`add`** - Add N to all values (hyphens unchanged)
```
(add (seq 1 2 3) 10)           → 11 12 13
(add (seq 60 64 67) 12)        → 72 76 79  (transpose octave)
```

**`mul`** - Multiply all values by N
```
(mul (seq 1 2 3) 2)            → 2 4 6
```

**`mod`** - Modulo all values by N
```
(mod (seq 8 9 10 11) 7)        → 1 2 3 4
```

**`scale`** - Map values from one range to another
```
(scale (seq 0 1 2 3) 0 3 60 72)  → 60 64 68 72
```

**`clamp`** - Limit values to min/max range
```
(clamp (seq 1 5 10 15) 3 12)   → 3 5 10 12
```

### Rhythm Generators (3)

**`euclid`** - Euclidean rhythm with optional rotation
```
(euclid 3 8)                   → - - 1 - - 1 - 1
(euclid 5 8)                   → - 1 - 1 1 - 1 1
(euclid 3 8 2)                 → 1 - - 1 - 1 - -  (rotated by 2)

Pattern mode - distribute sequence values euclideanly:
(euclid (seq 60 64 67) 8)      → - - 60 - - 64 - 67
(euclid (chord 60 0) 16)       → - 60 - - 64 - - 67 - - 60 - - 64 - -
(euclid (range 0 4) 16)        → - - - 0 - - 1 - - 2 - - 3 - - 4
```

**`bjork`** - Björklund algorithm distribution
```
(bjork 3 8)                    → 1 - - 1 - - 1 -
(bjork 5 8)                    → 1 - 1 - 1 - 1 1
```

**`subdiv`** - Subdivide each element N times
```
(subdiv (seq 1 2 3) 2)         → 1 1 2 2 3 3
(subdiv (seq 1 - 2) 3)         → 1 1 1 - - - 2 2 2
```

### List Manipulation (6)

**`reverse`** - Reverse sequence order
```
(reverse (seq 1 2 3 4))        → 4 3 2 1
```

**`rotate`** - Rotate sequence (positive = right, negative = left)
```
(rotate (seq 1 2 3 4) 1)       → 4 1 2 3
(rotate (seq 1 2 3 4) -1)      → 2 3 4 1
```

**`palindrome`** - Create palindrome (forward then backward, no pivot duplication)
```
(palindrome (seq 1 2 3))       → 1 2 3 2 1
```

**`mirror`** - Append reversed version (with pivot)
```
(mirror (seq 1 2 3))           → 1 2 3 3 2 1
```

**`interleave`** - Zip two sequences alternating
```
(interleave (seq 1 2 3) (seq - - -))  → 1 - 2 - 3 -
(interleave (seq 1 3 5) (seq 2 4 6))  → 1 2 3 4 5 6
```

**`shuffle`** - Randomize order (changes each evaluation)
```
(shuffle (seq 1 2 3 4))        → 3 1 4 2  (random)
```

### Selection & Filtering (5)

**`take`** - Take first N elements
```
(take (seq 1 2 3 4 5) 3)       → 1 2 3
```

**`drop`** - Drop first N elements
```
(drop (seq 1 2 3 4 5) 2)       → 3 4 5
```

**`every`** - Take every Nth element
```
(every (seq 1 2 3 4 5 6) 2)    → 1 3 5
(every (seq 1 2 3 4 5 6 7 8) 3) → 1 4 7
```

**`slice`** - Extract range [start, end)
```
(slice (seq 1 2 3 4 5) 1 4)    → 2 3 4
```

**`filter`** - Remove all hyphens
```
(filter (seq 1 - 2 - 3))       → 1 2 3
```

### Randomness & Probability (4)

**`choose`** - Randomly pick one argument
```
(choose (seq 1 2 3) (seq 4 5 6))  → randomly outputs one sequence
```

**`rand`** - Generate N random values between min and max
```
(rand 8 1 6)                   → 3 5 1 4 2 6 3 2  (random)
(rand 8 0 1)                   → 1 0 1 1 0 0 1 0  (binary)
```

**`prob`** - Each element has probability P (0-100) of appearing (else hyphen)
```
(prob (seq 1 2 3 4) 50)        → 1 - 3 -  (50% chance each)
```

**`maybe`** - Choose between two sequences with probability P (0-100)
```
(maybe (seq 1 2 3) (seq - - -) 70)  → 70% chance first, 30% second
```

**`degrade`** - Random degradation to hyphens
```
(degrade (seq 1 2 3 4) 25)     → 1 - 3 4  (25% become -)
```

### Pattern Generation (3)

**`range`** - Generate sequence from start to end with optional step
```
(range 1 5)                    → 1 2 3 4 5
(range 0 10 2)                 → 0 2 4 6 8 10
(range 10 0 -2)                → 10 8 6 4 2 0
```

**`ramp`** - Linear interpolation over N steps
```
(ramp 0 10 5)                  → 0 3 6 9 10
(ramp 60 72 7)                 → 60 62 64 66 68 70 72
```

**`drunk`** - Random walk (steps, max_step_size, start)
```
(drunk 8 2 60)                 → 60 61 62 64 63 61 62 64  (random walk)
```

### Conditional & Logic (2)

**`cycle`** - Repeat pattern to length N
```
(cycle (seq 1 2 3) 8)          → 1 2 3 1 2 3 1 2
```

**`grow`** - Progressive reveal (unmask pattern step by step)
```
(grow (seq 1 2 3))             → 1 - - 1 2 - 1 2 3
```

### Musical Operators (4)

**`transpose`** - Transpose notes (alias for add)
```
(transpose (seq 60 64 67) 5)   → 65 69 72
```

**`quantize`** - Snap to nearest scale degree
```
(quantize (seq 61 63 66) (seq 60 62 64 65 67 69 71 72))
  → 60 62 67  (quantized to C major)
```

**`chord`** - Generate chord from root and type
```
(chord 60 0)                   → 60 64 67     (C major)
(chord 60 1)                   → 60 63 67     (C minor)
(chord 60 4)                   → 60 64 67 71  (Cmaj7)
```

Chord types: 0=major, 1=minor, 2=dim, 3=aug, 4=major7, 5=minor7, 6=dom7

**`arp`** - Arpeggiate pattern
```
(arp (chord 60 0) 0 8)         → 60 64 67 60 64 67 60 64  (up)
(arp (chord 60 0) 1 8)         → 67 64 60 67 64 60 67 64  (down)
(arp (chord 60 0) 2 8)         → 60 64 67 64 60 64 67 64  (up-down)
```

Direction: 0=up, 1=down, 2=updown

### Time & Phase (2)

**`delay`** - Prepend N hyphens (rhythmic delay)
```
(delay (seq 1 2 3) 2)          → - - 1 2 3
```

**`gate`** - Keep every Nth, rest become hyphens
```
(gate (seq 1 2 3 4 5 6) 2)     → 1 - 3 - 5 -
(gate (seq 1 2 3 4 5 6) 3)     → 1 - - 4 - -
```

## Live Coding Examples

### Euclidean Rhythm with Notes
```
(euclid (seq 60 64 67 72) 8)
  → - - 60 - - 64 - 67  (distributes notes euclideanly)

(euclid (chord 60 0) 16)
  → - 60 - - 64 - - 67 - - 60 - - 64 - -  (cycling chord pattern)
```

### Random Walk Quantized to Scale
```
(quantize (drunk 8 3 60) (seq 60 62 64 65 67 69 71 72))
  → 60 62 64 62 65 67 69 67  (C major scale)
```

### Euclidean Rhythm Variants
```
(seq (euclid 3 8) (euclid 5 8) (euclid 7 8))
  → complete polyrhythmic pattern
```

### Arpeggio with Delay
```
(delay (arp (chord 60 0) 0 8) 4)
  → - - - - 60 64 67 60 64 67 60 64
```

### Growing Pattern
```
(grow (seq 1 2 3 4))
  → 1 - - - 1 2 - - 1 2 3 - 1 2 3 4
```

### Probabilistic Variation
```
(maybe
  (seq (euclid 3 8) (range 60 67))
  (seq (bjork 5 8) (reverse (range 60 67)))
  50)
```

## Technical Notes

- **Hyphens** represent rests/silence and output as -1 in Pure Data
- **Numbers** are integers (MIDI notes, step values, etc.)
- **Nesting** is fully supported - operators can be nested arbitrarily deep
- **Memory** is managed automatically - no leaks
- **Randomness** uses system time seed, changes each evaluation
- **Errors** are reported to Pure Data console with descriptive messages

## Extending

To add new operators:

1. Add `NODE_XXX` to `NodeType` enum (alien.c:44)
2. Implement `eval_xxx()` function
3. Add to parser switch statement (alien.c:496)
4. Add test cases (seqgen.c for standalone)
5. Document here

## Files

- `alien.c` - Pure Data external source
- `alien.pd_darwin` - Compiled PD external (macOS)
- `alien-help.pd` - Pure Data help patch
- `seqgen.c` - Standalone CLI version
- `seqgen` - Compiled CLI tool
- `ALIEN-README.md` - This file

## Credits

Pattern language design inspired by TidalCycles, SuperCollider, and live coding communities.
