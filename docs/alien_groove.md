# alien_groove

Rhythmic pattern constrainer. Takes an input pattern and a template groove, outputs a pattern constrained to the template with variable strictness. Supports phase shifting for Reich-style phasing processes.

## Creation

```
[alien_groove]
```

Left inlet: pattern to constrain (hot). Right inlet: template pattern (cold).

## Messages

| Message | Example | Description |
|---------|---------|-------------|
| `strictness` | `strictness 75` | How strictly to enforce template (0-100, default 100) |
| `mode` | `mode pull` | Constraint mode: `mask`, `pull`, or `push` |
| `phase` | `phase 3` | Rotate template by N steps |

## Template

Send a list to the right inlet (or use the `template` message). Any non-zero number is a hit, zero or `-` is a rest.

```
template 1 0 0 1 0 0 0 1
```

Or from alien DSL output:

```
[alien]
|
[alien_groove] <- template inlet
```

## Modes

### mask (default)

Silence hits that don't align with the template. Each non-aligned hit has a `strictness`% chance of being silenced.

- strictness 100: only template-aligned hits survive
- strictness 0: everything passes through unchanged

### pull

Pull hits toward the nearest template beat position. Hits on template positions stay put. Off-template hits are probabilistically moved to the nearest template hit.

### push

Push hits away from template positions (counter-rhythm). Hits on template beats are moved to the nearest rest position. Creates rhythms that fill the gaps of the template.

## Phase shifting

The `phase` message rotates the template pattern by N steps before applying constraints. This is the tool for Reich-style phasing where two versions of the same pattern gradually shift against each other.

```
phase 0       -> template as-is
phase 1       -> template rotated 1 step
phase -2      -> template rotated -2 steps
```

Template `1 - - 1 - - - 1` with `phase 1` becomes `- - 1 - - - 1 1`.

Phase wraps around the template length and accepts any integer including negative values.

### Phasing process example

Increment phase over time to create a gradual phase shift:

```
[metro 500]
|
[counter]         <- counts 0, 1, 2, 3...
|
[pack phase 0]
|
[alien_groove]
```

Two instances of `[alien_groove]` with the same template but different phase values will produce Steve Reich-style phasing.

## Strictness

Controls the probability of enforcement. At 100 (default), the mode is fully applied. At 0, the input passes through unchanged. Values in between create probabilistic variation.

```
strictness 100    -> full enforcement
strictness 50     -> half the non-aligned hits survive
strictness 0      -> no effect (passthrough)
```
