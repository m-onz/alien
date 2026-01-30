# alien_scale

Scale and mode quantizer. Takes numbers in, outputs MIDI pitches constrained to a scale. Two input modes: **degree** (scale degree index) and **quantize** (snap any MIDI note to the nearest in-scale pitch).

Output is octave-folded into a playable MIDI range (default 40-79).

## Creation

```
[alien_scale]                      C major, degree mode
[alien_scale 60 dorian]            root=60, dorian mode
[alien_scale 48 minor quantize]    root=48, minor, quantize mode
```

## Messages

| Message | Example | Description |
|---------|---------|-------------|
| `root` | `root 48` | Set root note (MIDI pitch) |
| `scale` | `scale dorian` | Set scale by name |
| `scale` | `scale 0 2 4 5 7 9 11` | Set scale by intervals |
| `mode` | `mode quantize` | Input mode: `degree` or `quantize` |
| `degrees` | `degrees 0 2 4` | Set active degrees (harmonic field) |
| `degrees` | `degrees off` | Disable degree filter |
| `ref` | `ref 48` | Set reference pitch for interval filter |
| `intervals` | `intervals 0 5 7` | Set allowed intervals from reference |
| `intervals` | `intervals off` | Disable interval filter |
| `range` | `range 36 72` | Set output MIDI range |

## Built-in scales

**Modes:** major/ionian, dorian, phrygian, lydian, mixolydian, aeolian/minor, locrian

**Pentatonic:** pentatonic/pent, minpent

**Other:** blues, chromatic, wholetone, diminished, augmented

**Harmonic/melodic:** harmonic, melodic

**World:** hirajoshi, insen, iwato, hungarian, romanian, phrygdom

Or define your own by intervals: `scale 0 2 4 5 7 9 11`

## Quantize mode

In quantize mode, input MIDI notes are snapped to the nearest pitch in the current scale across all octaves. This is the mode you want when feeding output from `[else/sequencer]` or other sources that produce MIDI note numbers.

```
mode quantize
```

Send 61 (C#) with C major scale -> outputs 60 (C) or 62 (D), whichever is nearer.

## Harmonic field control (degrees)

The `degrees` message limits which scale degrees are active. This is the main tool for controlling harmonic colour — triads, clusters, suspended voicings, quartal harmony.

In C major (degrees 0-6 = C D E F G A B):

```
degrees 0 2 4         -> C E G triad field
degrees 0 1 4         -> C D G sus2 field
degrees 0 3 4         -> C F G sus4 field
degrees 0 1 3 4       -> C D F G (Glass-like sus field)
degrees 0 1 2         -> C D E cluster
degrees 0 3 4 6       -> C F G B quartal voicing
degrees off           -> all degrees active
```

Notes that land on inactive degrees are snapped to the nearest active degree.

### Minimalist composition examples

Shift the harmonic field over time for process-based composition:

```
degrees 0 3 4         -> sus4 field (C F G)
degrees 0 1 3 4       -> add a 2nd (C D F G)
degrees 0 2 4         -> open to triad (C E G)
degrees 0 1 2 3       -> contract to cluster (C D E F)
degrees 0 3 4         -> resolve back to sus4
```

For dorian with stacked 4ths:

```
scale dorian
degrees 0 3 6         -> D G C (quartal stack)
degrees 0 2 3 6       -> D F G C
```

## Interval filter

The `intervals` message constrains output to specific intervals from a reference pitch. Use this for drone-relative harmony or controlling consonance/dissonance.

```
ref 48
intervals 0 5 7          -> only unisons, 4ths, 5ths from ref
intervals 0 3 4 7 8 9    -> consonant intervals only
intervals off             -> disable
```

Combine with degree filtering:

```
mode quantize
degrees 0 1 3 4           -> sus field
ref 48
intervals 0 5 7           -> only open intervals from the drone
```

## Processing pipeline

Every note passes through these stages in order:

1. **Input** — scale degree (degree mode) or MIDI note (quantize mode)
2. **Scale snap** — quantize to nearest scale tone (quantize mode only)
3. **Degree filter** — snap to nearest active degree if filtered
4. **Interval filter** — snap to nearest allowed interval from reference
5. **Range fold** — octave-fold into output range (default 40-79)

## Output range

Output is octave-folded (not clamped) into the MIDI range, so the pitch class is always preserved. Default is 40-79 (E2 to G5). Change with:

```
range 48 72               -> C3 to C5
range 36 84               -> C2 to C6
```

Range must span at least 12 semitones.
