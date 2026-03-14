# alien

![alien](/alien.png)

A **Lisp-like pattern language** for algorithmic music in [Pure Data](https://puredata.info). Write expressive patterns as S-expressions, evaluate them to lists, and feed them to sequencers.

```lisp
(euclid (seq 60 64 67) 8)  →  60 - - 64 - - 67 -
```

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

## Pattern Language

### Basics

```lisp
60                    ; number (MIDI note)
-                     ; rest (also . or _)
(seq 1 2 3)           ; sequence
(rep (seq 1 2) 4)     ; repeat: 1 2 1 2 1 2 1 2
```

### Rhythm

```lisp
(euclid 3 8)                    ; Euclidean: 1 - - 1 - - 1 -
(euclid (seq 60 64 67) 8)       ; with melody: 60 - - 64 - - 67 -
(euclid 5 16 2)                 ; with rotation
(subdiv (seq 1 2) 3)            ; subdivide: 1 1 1 2 2 2
```

### Transform

```lisp
(reverse (seq 1 2 3))           ; 3 2 1
(rotate (seq 1 2 3 4) 1)        ; 2 3 4 1
(interleave (seq 1 2) (seq 3 4)); 1 3 2 4
(shuffle (seq 1 2 3 4))         ; random order
```

### Selection

```lisp
(take (seq 1 2 3 4 5) 3)        ; 1 2 3
(drop (seq 1 2 3 4 5) 2)        ; 3 4 5
(slice (seq 1 2 3 4 5) 1 3)     ; 2 3
(every (seq 1 2 3 4 5 6) 2)     ; 1 3 5
(filter (seq 1 - 2 - 3))        ; 1 2 3 (remove rests)
```

### Arithmetic

```lisp
(add (seq 60 62 64) 12)         ; 72 74 76
(sub (seq 72 76 79) 12)         ; 60 64 67
(mul (seq 1 2 3) 2)             ; 2 4 6
(mod (seq 10 11 12) 12)         ; 10 11 0
(scale (seq 0 64 127) 0 127 60 72)  ; map range
(clamp (seq 50 60 70 80) 55 75) ; 55 60 70 75
(wrap (seq 60 72 84) 60 72)     ; 60 60 60
(fold (seq 50 60 70 80) 55 75)  ; 60 60 70 70
```

### Random

```lisp
(choose 60 64 67)               ; pick one
(rand 4)                        ; 4 random values 0-127
(rand 4 60 72)                  ; 4 random values 60-72
(prob (seq 1 2 3 4) 50)         ; 50% chance each
(drunk 8 2 60)                  ; random walk: 8 steps, ±2, start 60
(drunk 8 3 60 48 72)            ; bounded random walk
```

### Musical

```lisp
(quantize (seq 61 63 66) (seq 0 2 4 5 7 9 11))  ; snap to scale
(arp (seq 60 64 67) 0 8)        ; arpeggiate up, 8 steps
(arp (seq 60 64 67) 1 8)        ; arpeggiate down
(arp (seq 60 64 67) 2 8)        ; arpeggiate up-down
```

### Structure

```lisp
(cycle (seq 1 2 3) 8)           ; 1 2 3 1 2 3 1 2
(grow (seq 1 2 3 4))            ; 1, 1 2, 1 2 3, 1 2 3 4
(gate (seq 1 2 3 4) 2)          ; 1 - 2 - 3 - 4 -
```

### Generation

```lisp
(range 1 5)                     ; 1 2 3 4 5
(range 0 10 2)                  ; 0 2 4 6 8 10
(ramp 60 72 5)                  ; 60 63 66 69 72
```

---

## Composition

Patterns nest freely:

```lisp
(euclid (arp (seq 60 64 67) 0 4) 16)
(interleave (euclid 3 8) (euclid 5 8))
(add (shuffle (seq 60 64 67 71)) (choose 0 12))
```

---

## CLI Tool

Test patterns without Pd:

```bash
./alien_parser '(euclid 5 8)'
./alien_parser '(seq 60 64 67)'
./alien_parser '(drunk 16 3 60)'
./alien_parser --test
```

---

## Examples

### Four-on-the-floor

```
[; kick (euclid 1 16)]
[; snare (seq - - - - 1 - - - - - - - 1 - - -)]
[; hihat (rep 1 16)]
```

### Generative melody

```
[; lead (fold (drunk 16 3 60) 48 72)]
```

### Polyrhythm

```
[; a (euclid 3 8)]
[; b (euclid 5 8)]
[; c (euclid 7 8)]
```

---

## Theme

The `theme/` folder contains a dark canvas theme.

## Credits

Named after the [Lisp alien](https://lispers.org/)

## License

MIT
