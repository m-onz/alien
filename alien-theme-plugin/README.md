# alien-theme-plugin

A dark-canvas GUI theme plugin for [Pure Data](https://puredata.info), designed to
accompany the [`alien`](https://github.com/m-onz/alien) library.

It is a pure Tcl GUI plugin — no binaries, so a single package works on every OS, CPU,
and Pd float size.

## Install

- In Pd: **Tools → Find externals**, search `alien-theme-plugin`, install.
- **Restart Pd.** GUI plugins are only loaded at startup — nothing happens until you do.

## How Pd finds it

Pd discovers GUI plugins by globbing `*-plugin/*-plugin.tcl` and `*-plugin.tcl` on its
search path. Both the folder (`alien-theme-plugin/`) and the file
(`alien-theme-plugin.tcl`) must keep the `-plugin` suffix or Pd will not load it.

## Manual install

Copy `alien-theme-plugin/` into `~/Documents/Pd/externals` (or any folder on Pd's search
path) and restart Pd.

## License

MIT — see [LICENSE](LICENSE).
