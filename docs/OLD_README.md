# alien 👽

![alien](/alien.png)

A Lisp-like pattern language for Pure Data. Write `(euclid 5 8)` and get `- 1 - 1 1 - 1 1`. Nest operators, generate rhythms, arpeggiate chords, quantize drunk walks to scales. Feed the output to `[else/sequencer]` and drive anything—synths, samplers, visuals, whatever.

## Why

The sequences that `[else/sequencer]` accepts—hyphens and numbers like `- - 2 3 - - 4`—are deceptively powerful. A metro steps through them. Hyphens are rests. Numbers trigger events. This simple format can drive any parameter in Pure Data: MIDI notes, filter cutoffs, GEM coordinates, video playback positions.

The problem is writing interesting sequences by hand. Euclidean rhythms, arpeggios, probability-based variations, drunk walks—you want these things but typing them out is tedious. alien gives you a language to describe patterns compositionally. Nest operators. Build complexity from simple pieces.

## What's in the box

**Pure Data externals:**
- `[alien]` — the pattern language interpreter
- `[alien_router]` — route messages by first symbol (like `[route]` but keeps the whole pattern)
- `[alien_scale]` — scale/mode quantizer with 20+ built-in scales
- `[alien_groove]` — constrain patterns to a template groove

**Abstractions:**
- `[alien_monosynth~]` / `[alien_monosynth2~]` — simple mono synths
- `[video]` — GEM video player
- `[playdir~]` — directory-based sample player

**Tools:**
- `alien_parser` — standalone CLI for testing patterns
- `alien_evolve_py/` — genetic pattern evolution (Python)

## Install

```bash
make
make install
```

Needs Pure Data, pd-else (via deken), and a C compiler. Tested on macOS, should work on Linux.

## Quick start

```
[symbol (euclid 5 8)(
|
[alien]
|
[print]
```

Output: `- 1 - 1 1 - 1 1`

From the command line:

```bash
./alien_parser "(euclid (chord 60 0) 16)"
# - 60 - - 64 - - 67 - - 60 - - 64 - -
```

## The language

39 operators. Nest them freely.

**Rhythm:** `euclid`, `bjork`, `subdiv`, `gate`

```lisp
(euclid 3 8)                    ; - - 1 - - 1 - 1
(euclid (seq 60 64 67) 8)       ; distribute notes euclideanly
(bjork 5 13)                    ; Bjorklund's algorithm
```

**Melody:** `chord`, `arp`, `range`, `transpose`, `quantize`

```lisp
(chord 60 0)                    ; C major triad: 60 64 67
(chord 60 1)                    ; C minor: 60 63 67
(arp (chord 60 4) 0 16)         ; arpeggiate Cmaj7 upward, 16 steps
(quantize (drunk 8 3 60) (range 60 72))  ; drunk walk snapped to chromatic
```

**Transform:** `reverse`, `palindrome`, `rotate`, `interleave`, `mirror`, `shuffle`

```lisp
(palindrome (seq 1 2 3))        ; 1 2 3 2 1
(interleave (euclid 3 8) (range 60 67))  ; weave rhythm with melody
(rotate (seq 1 2 3 4) 1)        ; 4 1 2 3
```

**Probability:** `prob`, `degrade`, `maybe`, `choose`

```lisp
(prob (seq 60 64 67) 50)        ; each note has 50% chance
(degrade (euclid 5 8) 25)       ; randomly silence 25% of hits
(maybe (seq 1 2 3) (seq 4 5 6) 70)  ; 70% first, 30% second
(choose (chord 60 0) (chord 65 1) (chord 67 0))  ; pick one randomly
```

**Structure:** `seq`, `rep`, `cycle`, `take`, `drop`, `slice`, `grow`

```lisp
(rep (seq 60 - 64 -) 4)         ; repeat 4 times
(cycle (seq 1 2 3) 8)           ; 1 2 3 1 2 3 1 2
(take (range 60 72) 4)          ; first 4 elements
```

**Arithmetic:** `add`, `mul`, `mod`, `scale`, `clamp`

```lisp
(add (seq 60 64 67) 12)         ; transpose up an octave
(scale (range 0 8) 0 8 60 72)   ; map 0-8 to MIDI 60-72
```

Full reference: `docs/OPERATORS.md`

## Pattern evolution

The `alien_evolve_py/` directory contains a genetic algorithm that breeds patterns. It uses n-gram models, mutation, crossover, and novelty-based fitness to generate increasingly interesting valid patterns.

```bash
cd alien_evolve_py
python3 alien_evolve.py
```

Each run:
- Loads existing patterns from `patterns.txt`
- Evolves 100 generations
- Filters for diversity and proper rest distribution
- Appends novel patterns back to `patterns.txt`

The corpus grows over time. Use it as a pattern bank in your patches.

TODO & COMING SOON: alien_pattern_bank to play pre-computed / pre-evolved patterns.

## Audio/visual notes

For GEM visuals with audio, you may need to increase the audio block size to 1024 and delay to 256µs in Pd's audio settings. External sound cards help. Don't delete GEM objects without a window open on macOS—Pd will crash.

The `example-av-patch.pd` shows a working setup:
- Framebuffer for applying pixel effects to the final output
- Video playback from a folder
- GLSL fragment shaders (see `shader/` directory)
- GEM 3D primitives

The patch doesn't prescribe how to map audio parameters to visual ones or vice versa—that's your call. The right mapping depends on what you're making. Copy the example, swap in your own sounds and videos, wire things up how you want.

## Adding operators

1. Add `NODE_XXX` to `NodeType` in `alien_core.h`
2. Write `eval_xxx()` function
3. Add to parser switch
4. Add tests to `alien_parser.c`
5. Document in `docs/OPERATORS.md`

## License

MIT

## Credits

Named after the [Lisp alien](https://lispers.org/). Inspired by TidalCycles, SuperCollider, and the live coding community.
