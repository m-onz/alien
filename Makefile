#!/usr/bin/make -f
# Makefile for the 'alien' library for Pure Data.
# Uses pd-lib-builder: https://github.com/pure-data/pd-lib-builder

lib.name = alien

# Default install destination for a bare `make install`. pd-lib-builder's
# per-platform default (e.g. ~/Library/Pd on macOS, /usr/local/lib/pd-externals
# on Linux) is not where most users keep their patches, so point at the common
# ~/Documents/Pd/externals folder instead. This is only a default: an
# `objectsdir` passed on the command line (as the CI does with
# `objectsdir=./build`) always overrides it. Do NOT use `sudo` for this — sudo
# sets HOME=/var/root and the files land in root's home instead of yours.
objectsdir ?= $(HOME)/Documents/Pd/externals

# Class name == source file basename. Sources may live in subdirectories;
# pd-lib-builder still emits every binary flat in the repo root, which is
# exactly what a deken package wants. The standalone CLI tools
# (alien_parser.c, novelty/ns_parser.c, novelty/ns_system_test.c,
# novelty/validate_seeds.c) are NOT Pd externals and are intentionally
# excluded from this build.
class.sources = \
    alien.c \
    alien_wrap.c \
    alien_join.c \
    alien_snap.c \
    alien_scale.c \
    alien_cue.c \
    alien_orchestrate.c \
    novelty/ns_archive.c \
    novelty/ns_mutate.c \
    novelty/ns_log.c \
    novelty/ns_grid_stats.c \
    novelty/ns_spigot.c \
    novelty/ns_seq_features.c \
    novelty/ns_seq_propose.c \
    novelty/ns_seq_info.c \
    novelty/ns_quality.c \
    novelty/ns_corpus.c \
    novelty/ns_ast_features.c

# Installed FLAT into the package root. Help patches MUST be here (not in a
# subfolder) or deken will not find them when building the object list.
datafiles = \
    alien-meta.pd \
    alien-help.pd \
    alien_wrap-help.pd \
    alien_join-help.pd \
    alien_snap-help.pd \
    alien_scale-help.pd \
    alien_cue-help.pd \
    alien_orchestrate-help.pd \
    ns_archive-help.pd \
    ns_mutate-help.pd \
    ns_log-help.pd \
    ns_grid_stats-help.pd \
    ns_spigot-help.pd \
    ns_seq_features-help.pd \
    ns_seq_propose-help.pd \
    ns_seq_info-help.pd \
    ns_quality-help.pd \
    ns_corpus-help.pd \
    ns_ast_features-help.pd \
    novelty/ns-help.pd \
    novelty/novelty_engine.pd \
    pkg-tester.pd \
    README.md \
    LICENSE \
    alien.png

# Prefer the git submodule; fall back to the vendored copy so that source
# tarballs and the deken (Sources) package still build.
PDLIBBUILDER = $(firstword $(wildcard \
    pd-lib-builder/Makefile.pdlibbuilder \
    Makefile.pdlibbuilder))
include $(PDLIBBUILDER)
