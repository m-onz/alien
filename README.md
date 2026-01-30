
# alien 👽

![alien](/alien.png)

An algorithmic audio visual toolkit for [pure data and GEM](https://puredata.info)

# Get started

To use alien you will need:

* Install vanilla [pure data and GEM](https://puredata.info/downloads/pure-data)

You might not want pd-extended, purr-data. I recommend using Pd vanilla and adding dependencies as needed.

* download or clone this repository

With gcc or a c compiler

```bash
cd alien
make
make install
```

* within Pd add the alien folder to the path
* via deken (tools/find externals) install pd-else, freeverb~ & potentially GEM
* if you want inverted patch cables check out the "theme" folder and copy it to your Pd externals folder

# Workflow

Once you have installed everything you should be able to open a new pd patch and create an [alien] object. If you see dashed lines around the edge it means you have not successfully added alien to your Pd path. The Pd path is NOT the same thing as linux environment variables.

Copy the "alien_template" anywhere on your computer and rename it.

That is is how you can get Pd with batteries included and have a powerful algorithmic audio visual starting point

Add more sounds and video's to the alien folder, or reference sounds or artifacts in your local folder.

# What's in the box

**Pure Data externals:**
- `[alien]` — the pattern language interpreter
- `[alien_router]` — route messages by first symbol
- `[alien_scale]` — scale/mode quantizer with harmonic field control
- `[alien_groove]` — rhythmic pattern constrainer with phase shifting

**Abstractions:**
- `[alien_visuals]` — vj rig for adding pixel effects to the framebuffer
- `[alien_monosynth~]` / `[alien_monosynth2~]` — simple mono synths
- `[video]` — GEM video player
- `[playdir~]` — directory-based sample player

# Documentation

See the [docs](docs/) folder:

- [Alien DSL operators](docs/OPERATORS.md) — full reference for the pattern language
- [alien_scale](docs/alien_scale.md) — scale quantizer, harmonic fields, interval filters
- [alien_groove](docs/alien_groove.md) — rhythmic constrainer, phase shifting
- alien_evolve - experimental pattern evolution

# Credits

Named after the [Lisp alien](https://lispers.org/). Inspired by TidalCycles, SuperCollider, and the live coding community.
