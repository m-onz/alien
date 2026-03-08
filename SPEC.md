# Alien Pattern Language Specification

**Version 1.0** | March 2026

---

## Abstract

Alien is a domain-specific language (DSL) for algorithmic pattern generation in music and multimedia applications. It provides a minimal, composable syntax for expressing rhythmic, melodic, and structural patterns as S-expressions.

---

## 1. Introduction

### 1.1 Purpose

Alien enables musicians, live coders, and multimedia artists to express complex musical patterns concisely. The language prioritizes:

- **Composability** — patterns nest arbitrarily
- **Brevity** — common operations have short names
- **Determinism** — same input produces same output (except random ops)
- **Portability** — pure functional semantics, no side effects

### 1.2 Scope

This specification defines:

- Lexical structure
- Syntax grammar
- Operator semantics
- Type system
- Evaluation rules

### 1.3 Conformance

A conforming implementation MUST implement all operators defined in this specification with the specified semantics.

---

## 2. Lexical Structure

### 2.1 Character Set

Alien source text is encoded in UTF-8. Only ASCII characters are significant to the lexer.

### 2.2 Tokens

```
TOKEN     ::= LPAREN | RPAREN | NUMBER | HYPHEN | SYMBOL
LPAREN    ::= '('
RPAREN    ::= ')'
NUMBER    ::= '-'? [0-9]+ ('.' [0-9]+)?
HYPHEN    ::= '-'
SYMBOL    ::= [a-zA-Z_][a-zA-Z0-9_]*
WHITESPACE::= [ \t\n\r]+
```

### 2.3 Comments

Comments are not part of the core specification. Implementations MAY support line comments beginning with `;`.

---

## 3. Syntax

### 3.1 Grammar

```
expr       ::= atom | list
atom       ::= NUMBER | HYPHEN | SYMBOL
list       ::= '(' SYMBOL expr* ')'
program    ::= expr
```

### 3.2 Examples

```lisp
60                          ; number literal
-                           ; rest (hyphen)
(seq 1 2 3)                 ; operator application
(euclid (seq 60 64 67) 8)   ; nested expression
```

---

## 4. Type System

### 4.1 Values

Alien has two value types:

| Type | Description |
|------|-------------|
| `Integer` | Signed 32-bit integer |
| `Rest` | Absence of value (written `-`) |

### 4.2 Sequences

All expressions evaluate to **sequences** — ordered collections of values.

```
Sequence ::= (Integer | Rest)*
```

### 4.3 Internal Representation

- Integers are stored as `int32`
- Rests are represented as `-1` internally
- Empty sequences are valid: `()`

---

## 5. Operators

### 5.1 Notation

```
(operator arg1 arg2 ... argN) → result
```

Arguments in `[brackets]` are optional.

### 5.2 Sequence Construction

#### `seq` — Concatenate
```
(seq expr ...) → Sequence
```
Concatenates all arguments into a single sequence.

```lisp
(seq 1 2 3)           → 1 2 3
(seq 1 (seq 2 3) 4)   → 1 2 3 4
(seq)                 → (empty)
```

#### `rep` — Repeat
```
(rep pattern:Sequence n:Integer) → Sequence
```
Repeats `pattern` exactly `n` times.

```lisp
(rep (seq 1 2) 3)     → 1 2 1 2 1 2
(rep 60 4)            → 60 60 60 60
```

#### `range` — Integer Range
```
(range start:Integer end:Integer [step:Integer]) → Sequence
```
Generates integers from `start` to `end` inclusive. Default `step` is 1.

```lisp
(range 0 4)           → 0 1 2 3 4
(range 0 8 2)         → 0 2 4 6 8
(range 10 0 -2)       → 10 8 6 4 2 0
```

#### `ramp` — Linear Interpolation
```
(ramp start:Integer end:Integer steps:Integer) → Sequence
```
Generates `steps` values linearly interpolated from `start` to `end`.

```lisp
(ramp 0 100 5)        → 0 25 50 75 100
```

---

### 5.3 Rhythm

#### `euclid` — Euclidean Rhythm
```
(euclid hits:Integer steps:Integer [rotation:Integer]) → Sequence
(euclid pattern:Sequence steps:Integer [rotation:Integer]) → Sequence
```
Distributes `hits` evenly across `steps` using the Euclidean algorithm. If `pattern` is provided, distributes pattern values at hit positions.

```lisp
(euclid 3 8)              → 1 - - 1 - - 1 -
(euclid 5 8)              → 1 - 1 - 1 - 1 1
(euclid 3 8 2)            → - 1 - - 1 - - 1
(euclid (seq 60 64 67) 8) → 60 - - 64 - - 67 -
```

#### `bjork` — Bjorklund Algorithm
```
(bjork hits:Integer steps:Integer) → Sequence
```
Alternative Euclidean distribution using Bjorklund's algorithm.

#### `subdiv` — Subdivide
```
(subdiv pattern:Sequence n:Integer) → Sequence
```
Repeats each element `n` times.

```lisp
(subdiv (seq 1 2) 3)      → 1 1 1 2 2 2
```

---

### 5.4 Transformation

#### `reverse` — Reverse Order
```
(reverse pattern:Sequence) → Sequence
```

```lisp
(reverse (seq 1 2 3))     → 3 2 1
```

#### `rotate` — Rotate Elements
```
(rotate pattern:Sequence n:Integer) → Sequence
```
Rotates right by `n` positions. Negative `n` rotates left.

```lisp
(rotate (seq 1 2 3 4) 1)  → 4 1 2 3
(rotate (seq 1 2 3 4) -1) → 2 3 4 1
```

#### `palindrome` — Palindrome (Exclusive)
```
(palindrome pattern:Sequence) → Sequence
```
Appends reverse excluding last element.

```lisp
(palindrome (seq 1 2 3))  → 1 2 3 2 1
```

#### `mirror` — Mirror (Inclusive)
```
(mirror pattern:Sequence) → Sequence
```
Appends full reverse.

```lisp
(mirror (seq 1 2 3))      → 1 2 3 3 2 1
```

#### `interleave` — Interleave
```
(interleave a:Sequence b:Sequence) → Sequence
```
Alternates elements from `a` and `b`.

```lisp
(interleave (seq 1 2 3) (seq a b c)) → 1 a 2 b 3 c
```

#### `shuffle` — Random Shuffle
```
(shuffle pattern:Sequence) → Sequence
```
Returns elements in random order.

---

### 5.5 Selection

#### `take` — First N Elements
```
(take pattern:Sequence n:Integer) → Sequence
```

```lisp
(take (range 1 10) 3)     → 1 2 3
```

#### `drop` — Remove First N
```
(drop pattern:Sequence n:Integer) → Sequence
```

```lisp
(drop (range 1 5) 2)      → 3 4 5
```

#### `slice` — Subsequence
```
(slice pattern:Sequence start:Integer end:Integer) → Sequence
```
Extracts elements from index `start` to `end` (exclusive).

```lisp
(slice (range 0 9) 2 5)   → 2 3 4
```

#### `every` — Every Nth Element
```
(every pattern:Sequence n:Integer) → Sequence
```

```lisp
(every (range 1 8) 2)     → 1 3 5 7
```

#### `filter` — Remove Rests
```
(filter pattern:Sequence) → Sequence
```

```lisp
(filter (seq 1 - 2 - 3))  → 1 2 3
```

---

### 5.6 Arithmetic

#### `add` — Addition
```
(add pattern:Sequence n:Integer) → Sequence
```
Adds `n` to each non-rest value.

```lisp
(add (seq 60 64 67) 12)   → 72 76 79
```

#### `mul` — Multiplication
```
(mul pattern:Sequence n:Integer) → Sequence
```

```lisp
(mul (seq 1 2 3) 10)      → 10 20 30
```

#### `mod` — Modulo
```
(mod pattern:Sequence n:Integer) → Sequence
```

```lisp
(mod (range 0 7) 4)       → 0 1 2 3 0 1 2 3
```

#### `scale` — Range Mapping
```
(scale pattern:Sequence from_min from_max to_min to_max) → Sequence
```
Maps values from one range to another.

```lisp
(scale (range 0 4) 0 4 60 72) → 60 63 66 69 72
```

#### `clamp` — Constrain Range
```
(clamp pattern:Sequence min:Integer max:Integer) → Sequence
```
Constrains values to `[min, max]`.

```lisp
(clamp (range 0 10) 3 7)  → 3 3 3 3 4 5 6 7 7 7 7
```

#### `wrap` — Modulo Wrap
```
(wrap pattern:Sequence min:Integer max:Integer) → Sequence
```
Wraps values to range using modulo.

```lisp
(wrap (seq 60 72 84) 60 72) → 60 60 60
```

#### `fold` — Reflect at Boundaries
```
(fold pattern:Sequence min:Integer max:Integer) → Sequence
```
Reflects values at boundaries.

```lisp
(fold (seq 50 60 70 80) 55 70) → 60 60 70 60
```

---

### 5.7 Randomness

#### `choose` — Random Choice
```
(choose expr ...) → Sequence
```
Evaluates and returns one argument at random.

```lisp
(choose 60 64 67)         → 60 | 64 | 67
```

#### `rand` — Random Values
```
(rand count:Integer [min:Integer max:Integer]) → Sequence
```
Generates `count` random integers. Default range is 0-127 (MIDI).

```lisp
(rand 4)                  → (4 random values 0-127)
(rand 4 60 72)            → (4 random values 60-72)
```

#### `prob` — Probabilistic Filter
```
(prob pattern:Sequence percent:Integer) → Sequence
```
Keeps each value with `percent`% probability, else rest.

```lisp
(prob (seq 1 2 3 4) 50)   → (each kept ~50%)
```

#### `maybe` — Conditional Choice
```
(maybe a:Sequence b:Sequence percent:Integer) → Sequence
```
Returns `a` with `percent`% probability, else `b`.

```lisp
(maybe (seq 1 2) (seq 3 4) 75) → (seq 1 2) 75% | (seq 3 4) 25%
```

#### `drunk` — Random Walk
```
(drunk steps:Integer max_step:Integer start:Integer [min max]) → Sequence
```
Generates a random walk. Optional `min`/`max` bounds with reflection.

```lisp
(drunk 8 2 60)            → (8-step walk from 60, ±2)
(drunk 8 2 60 48 72)      → (bounded to 48-72)
```

#### `degrade` — Probabilistic Degradation
```
(degrade pattern:Sequence percent:Integer) → Sequence
```
Replaces values with rests at `percent`% probability.

```lisp
(degrade (range 1 8) 30)  → (~30% become rests)
```

---

### 5.8 Musical

#### `transpose` — Transposition
```
(transpose pattern:Sequence n:Integer) → Sequence
```
Alias for `add`. Transposes by `n` semitones.

#### `quantize` — Scale Quantization
```
(quantize pattern:Sequence scale:Sequence) → Sequence
```
Snaps each value to nearest value in `scale`.

```lisp
(quantize (rand 8 0 24) (seq 0 2 4 5 7 9 11))
```

#### `chord` — Chord Generation
```
(chord root:Integer type:Integer) → Sequence
```
Generates chord tones.

| Type | Name | Intervals |
|------|------|-----------|
| 0 | Major | 0 4 7 |
| 1 | Minor | 0 3 7 |
| 2 | Diminished | 0 3 6 |
| 3 | Augmented | 0 4 8 |
| 4 | Major 7th | 0 4 7 11 |
| 5 | Minor 7th | 0 3 7 10 |
| 6 | Dominant 7th | 0 4 7 10 |

```lisp
(chord 60 0)              → 60 64 67
(chord 60 4)              → 60 64 67 71
```

#### `arp` — Arpeggiator
```
(arp pattern:Sequence direction:Integer length:Integer) → Sequence
```
Arpeggiates pattern to `length` steps.

| Direction | Name |
|-----------|------|
| 0 | Up |
| 1 | Down |
| 2 | Up-Down |

```lisp
(arp (chord 60 0) 0 8)    → 60 64 67 60 64 67 60 64
(arp (chord 60 0) 2 8)    → 60 64 67 64 60 64 67 64
```

---

### 5.9 Structure

#### `cycle` — Cycle to Length
```
(cycle pattern:Sequence length:Integer) → Sequence
```
Repeats or truncates pattern to exact length.

```lisp
(cycle (seq 1 2 3) 8)     → 1 2 3 1 2 3 1 2
```

#### `grow` — Progressive Reveal
```
(grow pattern:Sequence) → Sequence
```
Outputs pattern with progressive reveal.

```lisp
(grow (seq 1 2 3))        → 1 - - 1 2 - 1 2 3
```

#### `delay` — Prepend Rests
```
(delay pattern:Sequence n:Integer) → Sequence
```
Prepends `n` rests.

```lisp
(delay (seq 1 2 3) 2)     → - - 1 2 3
```

#### `gate` — Sparse Gate
```
(gate pattern:Sequence n:Integer) → Sequence
```
Keeps every `n`th value, replaces others with rests.

```lisp
(gate (range 1 8) 2)      → 1 - 3 - 5 - 7 -
```

---

## 6. Evaluation

### 6.1 Order

Evaluation is **eager** and **left-to-right**. All arguments are evaluated before the operator is applied.

### 6.2 Rest Propagation

Arithmetic operators preserve rests:

```lisp
(add (seq 1 - 3) 10)      → 11 - 13
```

### 6.3 Error Handling

Implementations SHOULD report errors for:

- Unknown operators
- Incorrect argument counts
- Type mismatches
- Division by zero

---

## 7. Conformance Levels

### 7.1 Core

A **Core** implementation MUST support:
- `seq`, `rep`, `range`
- `euclid`, `subdiv`
- `reverse`, `rotate`
- `take`, `drop`
- `add`, `mul`, `mod`
- `choose`, `rand`
- `chord`, `arp`
- `cycle`, `delay`

### 7.2 Full

A **Full** implementation MUST support all operators in this specification.

---

## 8. References

- Toussaint, G. (2005). "The Euclidean Algorithm Generates Traditional Musical Rhythms"
- McLean, A. (2014). "Making Programming Languages to Dance to: Live Coding with Tidal"
- Collins, N. (2003). "Generative Music and Laptop Performance"

---

## Appendix A: Human-Friendly Syntax

### Note Names

Instead of MIDI numbers, use note names:

```lisp
(seq C4 E4 G4 C5)         → 60 64 67 72
(seq D#4 Bb3 F#5)         → 63 58 78
(chord C4)                → 60 64 67
```

| Format | Example | MIDI |
|--------|---------|------|
| Note + Octave | `C4` | 60 |
| Sharp | `C#4` | 61 |
| Flat | `Bb4` | 70 |
| Double sharp | `C##4` | 62 |
| Double flat | `Dbb4` | 60 |

### Rest Symbols

Multiple ways to write rests:

```lisp
(seq C4 - E4 - G4)        ; hyphen (original)
(seq C4 . E4 . G4)        ; dot
(seq C4 _ E4 _ G4)        ; underscore
```

All produce: `60 - 64 - 67`

### Hit Markers

Use `x` for a hit (value 1):

```lisp
(seq x . x . x . x .)     → 1 - 1 - 1 - 1 -
(euclid x 8)              → equivalent to (euclid 1 8)
```

### Chord Types

```lisp
(chord C4)                → 60 64 67 (major, default)
(chord C4 0)              → 60 64 67 (major)
(chord C4 1)              → 60 63 67 (minor)
(chord C4 4)              → 60 64 67 71 (maj7)
(chord C4 5)              → 60 63 67 70 (min7)
(chord C4 6)              → 60 64 67 70 (dom7)
```

| Type | Name | Intervals |
|------|------|-----------|
| 0 | Major | 0 4 7 |
| 1 | Minor | 0 3 7 |
| 2 | Diminished | 0 3 6 |
| 3 | Augmented | 0 4 8 |
| 4 | Major 7th | 0 4 7 11 |
| 5 | Minor 7th | 0 3 7 10 |
| 6 | Dominant 7th | 0 4 7 10 |
| 7 | Diminished 7th | 0 3 6 9 |
| 8 | Sus2 | 0 2 7 |
| 9 | Sus4 | 0 5 7 |

---

## Appendix B: MIDI Reference

| Note | MIDI | Note | MIDI |
|------|------|------|------|
| C3 | 48 | C4 | 60 |
| D3 | 50 | D4 | 62 |
| E3 | 52 | E4 | 64 |
| F3 | 53 | F4 | 65 |
| G3 | 55 | G4 | 67 |
| A3 | 57 | A4 | 69 |
| B3 | 59 | B4 | 71 |

---

## Appendix C: Scale Degrees

| Scale | Intervals |
|-------|-----------|
| Major | 0 2 4 5 7 9 11 |
| Minor | 0 2 3 5 7 8 10 |
| Dorian | 0 2 3 5 7 9 10 |
| Phrygian | 0 1 3 5 7 8 10 |
| Lydian | 0 2 4 6 7 9 11 |
| Mixolydian | 0 2 4 5 7 9 10 |
| Pentatonic | 0 2 4 7 9 |
| Blues | 0 3 5 6 7 10 |

---

**End of Specification**
