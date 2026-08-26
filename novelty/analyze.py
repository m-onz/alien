#!/usr/bin/env python3
"""Analyze patterns.txt — characterize the behavioral range of the archive."""

import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PARSER = os.path.abspath(os.path.join(HERE, "..", "alien_parser"))
PATTERNS = os.path.join(HERE, "patterns.txt")


def load():
    with open(PATTERNS) as f:
        return [line.strip().rstrip(";").strip() for line in f if line.strip()]


def run(src):
    try:
        r = subprocess.run([PARSER, src], capture_output=True, text=True, timeout=1.5)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    out = []
    for t in r.stdout.strip().split():
        if t == "-":
            out.append(None)
        else:
            try:
                out.append(int(t))
            except ValueError:
                return None
    return out


def ops_in(src):
    return re.findall(r"\(([a-z_]+)", src)


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def main():
    patterns = load()
    print(f"loaded {len(patterns)} patterns from {PATTERNS}")
    print()

    op_counts = Counter()
    depth_counts = Counter()
    source_lens = []
    out_lens = []
    rest_ratios = []
    uniq_pitches = []
    pitch_ranges = []
    interval_abs = []
    evals = 0
    failed = 0

    buckets = defaultdict(list)  # (rest_bin, range_bin) -> [pattern]

    for src in patterns:
        source_lens.append(len(src))
        ops = ops_in(src)
        op_counts.update(ops)
        depth_counts[src.count("(")] += 1

        seq = run(src)
        if seq is None:
            failed += 1
            continue
        evals += 1

        out_lens.append(len(seq))
        rests = sum(1 for v in seq if v is None)
        rr = rests / len(seq) if seq else 0
        rest_ratios.append(rr)
        notes = [v for v in seq if v is not None]
        uniq_pitches.append(len(set(notes)))
        if notes:
            pitch_ranges.append(max(notes) - min(notes))
        prev = None
        for v in seq:
            if v is None:
                continue
            if prev is not None:
                interval_abs.append(abs(v - prev))
            prev = v

        # Bucket for stratified sampling: rest ratio × pitch range.
        rb = min(4, int(rr * 5))        # 0..4
        pr = pitch_ranges[-1] if notes else 0
        pb = 0 if pr < 3 else 1 if pr < 8 else 2 if pr < 16 else 3
        buckets[(rb, pb)].append((src, seq))

    print(f"evaluated: {evals}  (failed: {failed})")
    print()
    print("=== operator usage (top 20) ===")
    total = sum(op_counts.values()) or 1
    for op, c in op_counts.most_common(20):
        print(f"  {op:12s} {c:5d}  {100*c/total:5.1f}%")

    print()
    print("=== source length ===")
    print(f"  mean {statistics.mean(source_lens):.1f}  median {statistics.median(source_lens):.0f}  "
          f"p90 {pct(source_lens,90)}  max {max(source_lens)}")

    print()
    print("=== AST size (paren count) ===")
    for d in sorted(depth_counts):
        print(f"  {d:2d} parens: {depth_counts[d]}")

    print()
    print("=== output length ===")
    print(f"  mean {statistics.mean(out_lens):.1f}  median {statistics.median(out_lens):.0f}  "
          f"p10 {pct(out_lens,10)}  p90 {pct(out_lens,90)}  max {max(out_lens)}")

    print()
    print("=== rest ratio (0 = all notes, 1 = all rests) ===")
    hist = [0] * 10
    for r in rest_ratios:
        hist[min(9, int(r * 10))] += 1
    for i, h in enumerate(hist):
        bar = "#" * int(40 * h / max(hist))
        print(f"  {i/10:.1f}-{(i+1)/10:.1f}: {h:4d} {bar}")

    print()
    print("=== unique pitches per output ===")
    print(f"  mean {statistics.mean(uniq_pitches):.1f}  median {statistics.median(uniq_pitches):.0f}  "
          f"p10 {pct(uniq_pitches,10)}  p90 {pct(uniq_pitches,90)}  max {max(uniq_pitches)}")

    print()
    print("=== pitch range per output ===")
    if pitch_ranges:
        print(f"  mean {statistics.mean(pitch_ranges):.1f}  median {statistics.median(pitch_ranges):.0f}  "
              f"p90 {pct(pitch_ranges,90)}  max {max(pitch_ranges)}")

    print()
    print("=== absolute interval distribution (distance between consecutive notes) ===")
    ivhist = Counter()
    for d in interval_abs:
        bin_ = min(24, d)
        ivhist[bin_] += 1
    total_iv = sum(ivhist.values()) or 1
    for d in sorted(ivhist):
        c = ivhist[d]
        bar = "#" * int(40 * c / max(ivhist.values()))
        label = f"{d}" if d < 24 else "24+"
        print(f"  {label:>3s} st: {c:5d}  {100*c/total_iv:5.1f}%  {bar}")

    print()
    print("=== stratified samples across behavior space ===")
    print("  (rest-ratio bin × pitch-range bin)")
    for (rb, pb), items in sorted(buckets.items()):
        print(f"\n  rests~{rb*20}-{rb*20+20}% | range~"
              f"{['tight','small','medium','wide'][pb]}  ({len(items)} patterns)")
        for src, seq in items[:2]:
            out = " ".join("-" if v is None else str(v) for v in seq[:20])
            suffix = " …" if len(seq) > 20 else ""
            print(f"    {src}")
            print(f"      → {out}{suffix}")


if __name__ == "__main__":
    main()
