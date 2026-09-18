# alien-theme-overlay-plugin

A GUI theme plugin for [Pure Data](https://puredata.info) that recolours the patch canvas
(cyan strokes/text, dark boxes) **and makes the empty canvas background transparent**, so a
[GEM](https://github.com/umlaeute/Gem) window running behind the patch shows through your
cables and boxes. Built for audio-visual performance: your visuals behind, your live patch
cables on top.

It is the transparent-background sibling of
[`alien-theme-plugin`](https://github.com/m-onz/alien): same recolouring, but the canvas is
see-through and the patch window floats on top.

It is a pure Tcl GUI plugin — no binaries, so a single package works on every OS, CPU and Pd
float size.

## Install

- In Pd: **Tools → Find externals**, search `alien-theme-overlay-plugin`, install.
- **Restart Pd.** GUI plugins are only loaded at startup — nothing happens until you do.
- Run **only one** of `alien-theme-overlay-plugin` / `alien-theme-plugin` at a time — both
  recolour the canvas.

## How Pd finds it

Pd discovers GUI plugins by globbing `*-plugin/*-plugin.tcl` and `*-plugin.tcl` on its
search path. Both the folder (`alien-theme-overlay-plugin/`) and the file
(`alien-theme-overlay-plugin.tcl`) must keep the `-plugin` suffix or Pd will not load it.

## Use

Just open patches — the canvas is transparent automatically and the window floats on top.
Make your GEM window fill the screen (e.g. a `fullscreen 1`, or `dimen`/`offset` messages to
`[gemwin]`), open your patch, and perform. The console prints
`alien-theme-overlay: loaded …` when it starts.

## Platform notes

Tk's transparency support differs by platform:

| platform | result |
|---|---|
| macOS (aqua) | true per-pixel transparency — the empty canvas is fully see-through |
| Windows / Linux | Tk has no per-pixel canvas transparency; the dark theme background is kept |

On macOS, clicks in the fully-transparent empty canvas may pass through to the GEM window
behind; clicks on objects and cables are unaffected.

## Manual install

Copy `alien-theme-overlay-plugin/` into `~/Documents/Pd/externals` (or any folder on Pd's
search path) and restart Pd.

## License

MIT — see [LICENSE](LICENSE).
