# alien 👽

![alien](/alien.png)

An algorithmic pattern language and externals for [Pure Data](https://puredata.info)

## Installation

```bash
cd alien
make
make install
```

Or specify a custom install path:

```bash
make install PREFIX=~/.pd-externals
```

Within Pd, add the alien folder to your path (File → Preferences → Path).

## Externals

| External | Description |
|----------|-------------|
| `[alien]` | Lisp-like pattern language interpreter |
| `[alien_router]` | Route messages by first symbol |
| `[alien_scale]` | Scale/mode quantizer with harmonic field control |
| `[alien_groove]` | Stochastic pattern filter with template masking |
| `[alien_cluster]` | Diatonic cluster chord generator with bass notes |

## CLI Tool

`alien_parser` — standalone command-line tool for testing patterns:

```bash
./alien_parser '(euclid 5 8)'
./alien_parser --test
```

## Theme

The `theme/` folder contains a dark canvas theme for Pd. Copy it to your Pd externals folder to use.

## Credits

Named after the [Lisp alien](https://lispers.org/). Inspired by TidalCycles, SuperCollider, and the live coding community.
