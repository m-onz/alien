
# alien 👽

![alien](/alien.png)

An algorithmic audio visual toolkit for [pure data and GEM](https://puredata.info)

# Get started

To use alien you will need: 

* Install vanilla [pure data and GEM](https://puredata.info/downloads/pure-data)

You do not want pd-extended, purr-data. It's better to use stable and original Pd vanilla and add dependencies as needed.

No one serious really cares about having curvy patch cables and all that guff.

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

# Alien DSL

Alien is built around a domain specific language for writing or live coding patterns. We send these patterns via messages (internally or externally via UDP updsend). The alien DSL has no concept of variables and other programming concepts that you might expect.

It's Lisp-like pattern language where you can write things lie `(euclid 5 8)` and get `- 1 - 1 1 - 1 1`. Nest operators, generate rhythms, arpeggiate chords, quantize drunk walks to scales. Feed the output to `[else/sequencer]` and drive anything—synths, samplers, visuals, whatever.

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
- `[alien_visuals]` — vj rig for adding pixel effects to the framebuffer
- `[alien_monosynth~]` / `[alien_monosynth2~]` — simple mono synths
- `[video]` — GEM video player
- `[playdir~]` — directory-based sample player

## Credits

Named after the [Lisp alien](https://lispers.org/). Inspired by TidalCycles, SuperCollider, and the live coding community.
