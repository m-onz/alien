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

Named after the [Lisp alien](https://lispers.org/)

## License

MIT
