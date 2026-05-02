#!/usr/bin/env python3
"""
novelty_search.py - Novelty search for Alien DSL patterns.

Inspired by Stanley & Lehman, "Why Greatness Cannot Be Planned."
There is no objective function. We maintain an archive of patterns whose
OUTPUT behavior (the evaluated sequence) is embedded as a feature vector.
A candidate is admitted iff its k-nearest-neighbor distance to the archive
exceeds a novelty threshold. The archive then seeds future mutation/crossover,
and the search drifts toward behavioral diversity rather than fitness.

Outputs are written to patterns.txt (one pattern per line, ending in `;`).
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ALIEN_PARSER = os.path.abspath(os.path.join(HERE, "..", "alien_parser"))
PATTERNS_FILE = os.path.join(HERE, "patterns.txt")

# Length constraints — reject runaway patterns early.
MAX_SOURCE_LEN = 140        # characters in the source expression
MAX_AST_NODES = 40          # total AST nodes
MAX_AST_DEPTH = 4           # nesting depth
MAX_OUTPUT_LEN = 96         # elements in the evaluated output
MIN_OUTPUT_LEN = 3          # too-short outputs are near-trivial

# Novelty search params.
K_NEAREST = 5               # k for k-NN novelty distance
NOVELTY_THRESHOLD = 0.18    # min mean k-NN distance for admission
ARCHIVE_CAP = 2000          # prune when it grows beyond this
PARSER_TIMEOUT = 1.5        # seconds per evaluation

# ---------------------------------------------------------------------------
# DSL operator specs — mirrors the C parser in alien_core.h.
# Args:
#   - "any"  : any expression (value or sub-pattern)
#   - "int"  : must be a literal number at this argument slot
#   - "seq"  : must be a sub-pattern (not a bare number/rest)
# (min_args, max_args, arg_types)
# ---------------------------------------------------------------------------

OPERATORS = {
    "seq":        (0, 8, ["any"]),
    "rep":        (2, 2, ["any", "int"]),
    "add":        (2, 2, ["seq", "int"]),
    "sub":        (2, 2, ["seq", "int"]),
    "mul":        (2, 2, ["seq", "int"]),
    "mod":        (2, 2, ["seq", "int"]),
    "scale":      (5, 5, ["seq", "int", "int", "int", "int"]),
    "clamp":      (3, 3, ["seq", "int", "int"]),
    "wrap":       (3, 3, ["seq", "int", "int"]),
    "fold":       (3, 3, ["seq", "int", "int"]),
    "euclid":     (2, 4, ["any", "int", "int", "int"]),
    "subdiv":     (2, 2, ["seq", "int"]),
    "reverse":    (1, 1, ["seq"]),
    "rotate":     (2, 2, ["seq", "int"]),
    "interleave": (2, 2, ["seq", "seq"]),
    "shuffle":    (1, 1, ["seq"]),
    "mirror":     (1, 1, ["seq"]),
    "take":       (2, 2, ["seq", "int"]),
    "drop":       (2, 2, ["seq", "int"]),
    "slice":      (3, 3, ["seq", "int", "int"]),
    "every":      (2, 2, ["seq", "int"]),
    "filter":     (1, 1, ["seq"]),
    "range":      (2, 3, ["int", "int", "int"]),
    "ramp":       (3, 3, ["int", "int", "int"]),
    "choose":     (2, 4, ["any"]),
    "rand":       (1, 3, ["int", "int", "int"]),
    "prob":       (2, 2, ["seq", "int"]),
    "drunk":      (3, 5, ["int", "int", "int", "int", "int"]),
    "quantize":   (2, 2, ["seq", "seq"]),
    "arp":        (3, 3, ["seq", "int", "int"]),
    "cycle":      (2, 2, ["seq", "int"]),
    "grow":       (1, 1, ["seq"]),
    "gate":       (2, 2, ["seq", "int"]),
    "speed":      (2, 2, ["seq", "int"]),
    "mask":       (2, 2, ["seq", "seq"]),
    "delay":      (2, 2, ["seq", "int"]),
}

ALL_OPS = list(OPERATORS.keys())

# Loose category tags — used for same-family mutation swaps.
OP_GROUPS = {
    "arith":   ["add", "sub", "mul", "mod"],
    "bound":   ["clamp", "wrap", "fold"],
    "rhythm":  ["euclid", "subdiv", "gate", "speed", "mask"],
    "list":    ["reverse", "rotate", "shuffle", "mirror"],
    "select":  ["take", "drop", "slice", "every", "filter"],
    "gen":     ["range", "ramp", "rand", "drunk"],
    "random":  ["choose", "prob", "rand", "drunk"],
    "time":    ["cycle", "delay", "gate", "speed"],
    "musical": ["quantize", "arp", "mirror"],
    "struct":  ["seq", "rep", "cycle", "grow"],
}


def group_of(op: str) -> str:
    for g, ops in OP_GROUPS.items():
        if op in ops:
            return g
    return "struct"


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Node:
    kind: str  # "num" | "rest" | operator name
    value: Optional[int] = None
    children: List["Node"] = field(default_factory=list)


def is_leaf(n: Node) -> bool:
    return n.kind in ("num", "rest")


def ast_render(n: Node) -> str:
    if n.kind == "num":
        return str(n.value)
    if n.kind == "rest":
        return "-"
    return "(" + n.kind + (" " + " ".join(ast_render(c) for c in n.children) if n.children else "") + ")"


def ast_size(n: Node) -> int:
    if is_leaf(n):
        return 1
    return 1 + sum(ast_size(c) for c in n.children)


def ast_depth(n: Node) -> int:
    if is_leaf(n):
        return 0
    if not n.children:
        return 1
    return 1 + max(ast_depth(c) for c in n.children)


def ast_copy(n: Node) -> Node:
    if is_leaf(n):
        return Node(kind=n.kind, value=n.value)
    return Node(kind=n.kind, children=[ast_copy(c) for c in n.children])


def all_subtrees(n: Node, acc=None) -> List[Tuple[Node, int]]:
    """Collect (node, depth) pairs for every subtree."""
    if acc is None:
        acc = []
    def walk(node, d):
        acc.append((node, d))
        for c in node.children:
            walk(c, d + 1)
    walk(n, 0)
    return acc


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------

def musical_int(kind: str = "any") -> int:
    """Pick an integer that's musically plausible in context."""
    if kind == "midi":
        return random.choice([36, 38, 42, 48, 52, 55, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 79])
    if kind == "steps":
        return random.choice([4, 6, 8, 12, 16])
    if kind == "hits":
        return random.choice([2, 3, 4, 5, 7])
    if kind == "small":
        return random.choice([1, 2, 3, 4, 5])
    if kind == "prob":
        return random.choice([20, 35, 50, 65, 80])
    if kind == "interval":
        return random.choice([-12, -7, -5, -3, 3, 5, 7, 12])
    # generic
    r = random.random()
    if r < 0.35: return musical_int("midi")
    if r < 0.55: return musical_int("steps")
    if r < 0.75: return musical_int("small")
    if r < 0.85: return musical_int("interval")
    return random.randint(0, 100)


def gen_leaf(rest_prob: float = 0.25) -> Node:
    if random.random() < rest_prob:
        return Node(kind="rest")
    return Node(kind="num", value=musical_int("midi"))


def gen_seq_literal(length: int = 4, rest_prob: float = 0.3) -> Node:
    return Node(kind="seq", children=[gen_leaf(rest_prob) for _ in range(length)])


def gen_scale_literal() -> Node:
    """Generate a likely-useful scale literal (pitch classes)."""
    scales = [
        [0, 2, 4, 5, 7, 9, 11],    # major
        [0, 2, 3, 5, 7, 8, 10],    # minor
        [0, 3, 5, 6, 7, 10],       # blues
        [0, 2, 4, 7, 9],           # pentatonic
        [0, 1, 3, 5, 6, 8, 10],    # locrian-ish
        [0, 4, 7],                 # triad
    ]
    degs = random.choice(scales)
    return Node(kind="seq", children=[Node(kind="num", value=d) for d in degs])


def gen_int_node(kind: str = "any") -> Node:
    return Node(kind="num", value=musical_int(kind))


def gen_op(op: str, depth: int) -> Node:
    """Generate a valid instance of `op`, filling children at the given depth budget."""
    def any_arg(remaining_depth: int) -> Node:
        if remaining_depth <= 0 or random.random() < 0.45:
            return gen_leaf(rest_prob=0.15)
        return gen_tree(remaining_depth - 1)

    def seq_arg(remaining_depth: int) -> Node:
        if remaining_depth <= 0 or random.random() < 0.35:
            return gen_seq_literal(random.randint(3, 6), rest_prob=0.25)
        return gen_tree(remaining_depth - 1, require_seq=True)

    rd = MAX_AST_DEPTH - depth  # depth budget remaining for children

    if op == "seq":
        n = random.randint(2, 6)
        return Node(kind="seq", children=[
            gen_leaf(0.3) if random.random() < 0.7 else any_arg(rd - 1)
            for _ in range(n)
        ])

    if op == "rep":
        return Node(kind="rep", children=[
            any_arg(rd - 1),
            Node(kind="num", value=random.choice([2, 3, 4, 6, 8, 16])),
        ])

    if op in ("add", "sub", "mul", "mod"):
        delta_kind = "interval" if op in ("add", "sub") else "small"
        val = musical_int(delta_kind)
        if op == "mod":
            val = random.choice([3, 4, 5, 7, 8, 12])
        return Node(kind=op, children=[seq_arg(rd - 1), Node(kind="num", value=val)])

    if op == "scale":
        fmin, fmax = 0, random.choice([10, 12, 100, 127])
        tmin = musical_int("midi")
        tmax = tmin + random.choice([7, 12, 24])
        return Node(kind="scale", children=[
            seq_arg(rd - 1),
            Node(kind="num", value=fmin),
            Node(kind="num", value=fmax),
            Node(kind="num", value=tmin),
            Node(kind="num", value=tmax),
        ])

    if op in ("clamp", "wrap", "fold"):
        lo = musical_int("midi") - random.choice([0, 4, 7])
        hi = lo + random.choice([7, 12, 24])
        return Node(kind=op, children=[
            seq_arg(rd - 1),
            Node(kind="num", value=lo),
            Node(kind="num", value=hi),
        ])

    if op == "euclid":
        # Either (hits steps) or (pattern steps) form; optional rotation/hit_value.
        if random.random() < 0.5:
            first = Node(kind="num", value=musical_int("hits"))
        else:
            first = seq_arg(rd - 1)
        steps = Node(kind="num", value=musical_int("steps"))
        children = [first, steps]
        if random.random() < 0.3:
            children.append(Node(kind="num", value=random.randint(0, 7)))
            if random.random() < 0.4:
                children.append(Node(kind="num", value=musical_int("midi")))
        return Node(kind="euclid", children=children)

    if op == "subdiv":
        return Node(kind="subdiv", children=[
            seq_arg(rd - 1),
            Node(kind="num", value=random.choice([2, 3, 4])),
        ])

    if op in ("reverse", "shuffle", "mirror", "filter", "grow"):
        return Node(kind=op, children=[seq_arg(rd - 1)])

    if op == "rotate":
        return Node(kind="rotate", children=[
            seq_arg(rd - 1),
            Node(kind="num", value=random.randint(1, 7)),
        ])

    if op == "interleave":
        return Node(kind="interleave", children=[seq_arg(rd - 1), seq_arg(rd - 1)])

    if op in ("take", "drop", "every", "gate", "speed", "delay", "cycle"):
        n_val = random.choice([2, 3, 4, 6, 8])
        if op == "every" or op == "gate":
            n_val = random.choice([2, 3, 4])
        return Node(kind=op, children=[
            seq_arg(rd - 1),
            Node(kind="num", value=n_val),
        ])

    if op == "slice":
        start = random.randint(0, 3)
        end = start + random.randint(2, 6)
        return Node(kind="slice", children=[
            seq_arg(rd - 1),
            Node(kind="num", value=start),
            Node(kind="num", value=end),
        ])

    if op == "range":
        start = musical_int("midi")
        end = start + random.choice([5, 7, 12, 24])
        children = [Node(kind="num", value=start), Node(kind="num", value=end)]
        if random.random() < 0.3:
            children.append(Node(kind="num", value=random.choice([1, 2, 3])))
        return Node(kind="range", children=children)

    if op == "ramp":
        start = musical_int("midi")
        end = start + random.choice([-12, -7, 7, 12, 24])
        return Node(kind="ramp", children=[
            Node(kind="num", value=start),
            Node(kind="num", value=end),
            Node(kind="num", value=musical_int("steps")),
        ])

    if op == "drunk":
        steps = musical_int("steps")
        mx = random.choice([1, 2, 3, 5])
        start = musical_int("midi")
        children = [Node(kind="num", value=steps),
                    Node(kind="num", value=mx),
                    Node(kind="num", value=start)]
        if random.random() < 0.5:
            lo = start - random.choice([5, 7, 12])
            hi = start + random.choice([5, 7, 12])
            children.append(Node(kind="num", value=lo))
            children.append(Node(kind="num", value=hi))
        return Node(kind="drunk", children=children)

    if op == "rand":
        count = musical_int("steps")
        children = [Node(kind="num", value=count)]
        if random.random() < 0.7:
            lo = musical_int("midi")
            hi = lo + random.choice([7, 12, 24])
            children.append(Node(kind="num", value=lo))
            children.append(Node(kind="num", value=hi))
        return Node(kind="rand", children=children)

    if op == "prob":
        return Node(kind="prob", children=[
            seq_arg(rd - 1),
            Node(kind="num", value=musical_int("prob")),
        ])

    if op == "choose":
        k = random.randint(2, 3)
        return Node(kind="choose", children=[any_arg(rd - 1) for _ in range(k)])

    if op == "quantize":
        return Node(kind="quantize", children=[seq_arg(rd - 1), gen_scale_literal()])

    if op == "arp":
        return Node(kind="arp", children=[
            seq_arg(rd - 1),
            Node(kind="num", value=random.randint(0, 2)),
            Node(kind="num", value=musical_int("steps")),
        ])

    if op == "mask":
        return Node(kind="mask", children=[seq_arg(rd - 1), seq_arg(rd - 1)])

    # Fallback — should not be reachable.
    return gen_seq_literal(4, 0.3)


def gen_tree(depth_budget: int = MAX_AST_DEPTH, require_seq: bool = False) -> Node:
    """Top-level random tree generator."""
    if depth_budget <= 0:
        return gen_seq_literal(4, 0.3) if require_seq else gen_leaf(0.15)
    # Prefer musically-productive roots.
    bias = [
        ("euclid", 4), ("seq", 3), ("interleave", 3), ("arp", 2),
        ("quantize", 2), ("mask", 2), ("rotate", 2), ("mirror", 2),
        ("drunk", 2), ("range", 2), ("ramp", 2), ("rep", 2),
        ("prob", 2), ("subdiv", 2), ("gate", 2), ("speed", 2),
        ("cycle", 2), ("grow", 1), ("reverse", 2), ("choose", 1),
        ("add", 1), ("mul", 1), ("mod", 1), ("scale", 1), ("clamp", 1),
        ("wrap", 1), ("fold", 1), ("take", 1), ("drop", 1), ("slice", 1),
        ("every", 1), ("filter", 1), ("rand", 1), ("delay", 1), ("shuffle", 1),
    ]
    ops, weights = zip(*bias)
    op = random.choices(ops, weights=weights, k=1)[0]
    return gen_op(op, MAX_AST_DEPTH - depth_budget)


# ---------------------------------------------------------------------------
# Mutation & crossover
# ---------------------------------------------------------------------------

def mutate(n: Node, rate: float = 0.2, depth: int = 0) -> Node:
    """Apply a mutation, recursing with decaying rate."""
    if random.random() > rate / (1 + 0.3 * depth):
        if is_leaf(n):
            return ast_copy(n)
        return Node(kind=n.kind, children=[mutate(c, rate, depth + 1) for c in n.children])

    # Mutate this node.
    if n.kind == "num":
        choice = random.random()
        if choice < 0.45:
            delta = random.choice([-12, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 12])
            return Node(kind="num", value=(n.value or 0) + delta)
        if choice < 0.6:
            return Node(kind="rest")
        if choice < 0.85:
            return Node(kind="num", value=musical_int())
        return gen_leaf(0.2)

    if n.kind == "rest":
        if random.random() < 0.5:
            return Node(kind="num", value=musical_int("midi"))
        return Node(kind="rest")

    # Operator node.
    choice = random.random()

    if choice < 0.3:
        # Recurse into children.
        return Node(kind=n.kind, children=[mutate(c, rate, depth + 1) for c in n.children])

    if choice < 0.45 and depth + 1 < MAX_AST_DEPTH:
        # Swap to a sibling operator in the same group.
        siblings = [o for o in OP_GROUPS.get(group_of(n.kind), []) if o != n.kind]
        if siblings:
            return gen_op(random.choice(siblings), depth)
        return Node(kind=n.kind, children=[mutate(c, rate, depth + 1) for c in n.children])

    if choice < 0.6:
        # Replace with a freshly-generated subtree.
        return gen_tree(MAX_AST_DEPTH - depth)

    if choice < 0.75:
        # Add/remove a child if the operator supports it.
        min_args, max_args, _ = OPERATORS[n.kind]
        kids = [mutate(c, rate, depth + 1) for c in n.children]
        if random.random() < 0.5 and (max_args is None or len(kids) < max_args):
            kids.append(gen_leaf(0.3))
        elif len(kids) > min_args:
            kids.pop(random.randrange(len(kids)))
        return Node(kind=n.kind, children=kids)

    if choice < 0.9:
        # Wrap this node in another operator.
        wrapper = random.choice(["reverse", "mirror", "shuffle", "cycle", "prob", "rep"])
        inner = ast_copy(n)
        if wrapper == "cycle":
            return Node(kind="cycle", children=[inner, Node(kind="num", value=musical_int("steps"))])
        if wrapper == "prob":
            return Node(kind="prob", children=[inner, Node(kind="num", value=musical_int("prob"))])
        if wrapper == "rep":
            return Node(kind="rep", children=[inner, Node(kind="num", value=random.choice([2, 3, 4]))])
        return Node(kind=wrapper, children=[inner])

    # Reorder children for order-sensitive operators.
    if len(n.children) >= 2:
        kids = [mutate(c, rate, depth + 1) for c in n.children]
        i, j = random.sample(range(len(kids)), 2)
        kids[i], kids[j] = kids[j], kids[i]
        return Node(kind=n.kind, children=kids)
    return Node(kind=n.kind, children=[mutate(c, rate, depth + 1) for c in n.children])


def crossover(a: Node, b: Node) -> Node:
    """Subtree crossover — graft a subtree from b into a random cut in a."""
    a_copy = ast_copy(a)
    donors = all_subtrees(b)
    # Walk `a_copy` and at one randomly selected internal/leaf node, splice a donor in.
    sites = all_subtrees(a_copy)
    if not sites or not donors:
        return a_copy
    target_node, target_depth = random.choice(sites)
    donor_node, donor_depth = random.choice(donors)
    # Overwrite target_node in place.
    new_sub = ast_copy(donor_node)
    target_node.kind = new_sub.kind
    target_node.value = new_sub.value
    target_node.children = new_sub.children
    return a_copy


# ---------------------------------------------------------------------------
# Validation via the external parser
# ---------------------------------------------------------------------------

def run_parser(source: str) -> Optional[str]:
    """Evaluate `source` with alien_parser. Returns stdout or None on error."""
    try:
        res = subprocess.run(
            [ALIEN_PARSER, source],
            capture_output=True, text=True, timeout=PARSER_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if res.returncode != 0 or res.stderr.strip():
        return None
    return res.stdout.strip()


def parse_output(text: str) -> List[Optional[int]]:
    """Parse parser stdout into a list where rests are None."""
    out: List[Optional[int]] = []
    for tok in text.split():
        if tok == "-":
            out.append(None)
            continue
        try:
            out.append(int(tok))
        except ValueError:
            return []
    return out


# ---------------------------------------------------------------------------
# Behavior embedding (feature vector for KNN novelty)
# ---------------------------------------------------------------------------

INTERVAL_BINS = [-24, -12, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 12, 24]


def embed(seq: List[Optional[int]]) -> List[float]:
    """Map an evaluated sequence to a fixed-dimension feature vector.

    Features (all normalized to roughly [0, 1]):
      - length (vs MAX_OUTPUT_LEN)
      - rest ratio
      - unique-note ratio
      - pitch mean (normalized to 0..127)
      - pitch range (normalized)
      - pitch std (normalized)
      - contour: fraction of ascending / descending / repeated intervals
      - rhythm entropy over run-lengths
      - lag-1 and lag-2 autocorrelation (of note-vs-rest mask)
      - interval histogram across INTERVAL_BINS (11+ dims)
    """
    n = max(1, len(seq))
    rest_ratio = sum(1 for v in seq if v is None) / n
    notes = [v for v in seq if v is not None]
    unique_ratio = (len(set(notes)) / len(notes)) if notes else 0.0

    if notes:
        mean = sum(notes) / len(notes)
        nmin, nmax = min(notes), max(notes)
        pitch_range = (nmax - nmin) / 127.0
        var = sum((x - mean) ** 2 for x in notes) / len(notes)
        std = math.sqrt(var) / 32.0
        mean_norm = max(0.0, min(1.0, mean / 127.0))
    else:
        mean_norm = 0.0
        pitch_range = 0.0
        std = 0.0

    # Intervals between consecutive *notes* (skipping rests).
    intervals: List[int] = []
    prev = None
    for v in seq:
        if v is None:
            continue
        if prev is not None:
            intervals.append(v - prev)
        prev = v

    if intervals:
        ascend = sum(1 for d in intervals if d > 0) / len(intervals)
        descend = sum(1 for d in intervals if d < 0) / len(intervals)
        repeat = sum(1 for d in intervals if d == 0) / len(intervals)
    else:
        ascend = descend = repeat = 0.0

    # Interval histogram — bucket each interval into the nearest bin.
    hist = [0.0] * len(INTERVAL_BINS)
    for d in intervals:
        idx = min(range(len(INTERVAL_BINS)), key=lambda i: abs(INTERVAL_BINS[i] - d))
        hist[idx] += 1.0
    total_hist = sum(hist)
    if total_hist > 0:
        hist = [h / total_hist for h in hist]

    # Rhythm run-length entropy — note/rest mask.
    mask = [0 if v is None else 1 for v in seq]
    if mask:
        runs: List[int] = []
        cur = mask[0]; count = 1
        for x in mask[1:]:
            if x == cur:
                count += 1
            else:
                runs.append(count); cur = x; count = 1
        runs.append(count)
        total = sum(runs)
        ent = 0.0
        for r in runs:
            p = r / total
            if p > 0:
                ent -= p * math.log2(p)
        # normalize by log2(len(runs)) ceiling
        denom = math.log2(max(2, len(runs)))
        rhythm_ent = ent / denom
    else:
        rhythm_ent = 0.0

    # Autocorrelation of mask at lags 1 and 2.
    def autocorr(lag: int) -> float:
        if len(mask) <= lag:
            return 0.0
        m = sum(mask) / len(mask)
        num = sum((mask[i] - m) * (mask[i + lag] - m) for i in range(len(mask) - lag))
        den = sum((x - m) ** 2 for x in mask) or 1.0
        return max(-1.0, min(1.0, num / den))

    ac1 = (autocorr(1) + 1) / 2  # map -1..1 → 0..1
    ac2 = (autocorr(2) + 1) / 2

    length_norm = min(1.0, n / MAX_OUTPUT_LEN)

    base = [
        length_norm, rest_ratio, unique_ratio,
        mean_norm, pitch_range, std,
        ascend, descend, repeat,
        rhythm_ent, ac1, ac2,
    ]
    return base + hist


def euclidean(a: List[float], b: List[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# Novelty archive
# ---------------------------------------------------------------------------

@dataclass
class ArchiveEntry:
    source: str
    feat: List[float]
    novelty: float
    added_at: float


class NoveltyArchive:
    def __init__(self, k: int = K_NEAREST, threshold: float = NOVELTY_THRESHOLD, cap: int = ARCHIVE_CAP):
        self.k = k
        self.threshold = threshold
        self.cap = cap
        self.entries: List[ArchiveEntry] = []
        self._seen: set = set()
        self._recent_added_frac: List[int] = []  # 1 if added, 0 if rejected — for adaptive threshold

    def knn_distance(self, feat: List[float]) -> float:
        if not self.entries:
            return float("inf")
        dists = [euclidean(feat, e.feat) for e in self.entries]
        dists.sort()
        top = dists[: min(self.k, len(dists))]
        return sum(top) / len(top)

    def consider(self, source: str, feat: List[float]) -> Tuple[bool, float]:
        if source in self._seen:
            return False, 0.0
        nov = self.knn_distance(feat)
        accept = nov >= self.threshold
        self._recent_added_frac.append(1 if accept else 0)
        if len(self._recent_added_frac) > 200:
            self._recent_added_frac.pop(0)
        if accept:
            self.entries.append(ArchiveEntry(source, feat, nov, time.time()))
            self._seen.add(source)
            if len(self.entries) > self.cap:
                # Drop the least-novel entry.
                i_min = min(range(len(self.entries)), key=lambda i: self.entries[i].novelty)
                dropped = self.entries.pop(i_min)
                self._seen.discard(dropped.source)
            self._adapt_threshold()
        return accept, nov

    def _adapt_threshold(self) -> None:
        """Aim for ~25% acceptance: raise bar when too many admits, lower when too few."""
        if len(self._recent_added_frac) < 50:
            return
        rate = sum(self._recent_added_frac) / len(self._recent_added_frac)
        if rate > 0.40:
            self.threshold *= 1.05
        elif rate < 0.10:
            self.threshold *= 0.95
        self.threshold = max(0.05, min(2.0, self.threshold))

    def sample(self, n: int) -> List[ArchiveEntry]:
        if n >= len(self.entries):
            return list(self.entries)
        return random.sample(self.entries, n)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

SEEDS = [
    "(seq 60 - 64 - 67 -)",
    "(euclid 3 8)",
    "(euclid 5 8)",
    "(euclid (seq 60 64 67) 8)",
    "(range 60 72)",
    "(arp (seq 60 64 67) 0 8)",
    "(mirror (seq 60 62 64 65 67))",
    "(drunk 16 3 60)",
    "(quantize (drunk 16 3 60) (seq 0 2 4 5 7 9 11))",
    "(mask (seq 60 64 67 72) (euclid 4 16))",
    "(interleave (euclid 3 8) (euclid 5 8))",
    "(rotate (seq 60 - 64 - 67) 2)",
    "(prob (seq 60 64 67 72) 50)",
    "(grow (seq 60 64 67 72))",
    "(gate (range 60 75) 3)",
    "(speed (euclid 3 8) 2)",
    "(fold (drunk 16 5 60) 55 70)",
    "(slice (cycle (seq 60 64 67 72) 32) 8 16)",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def append_patterns(paths: List[str]) -> None:
    if not paths:
        return
    with open(PATTERNS_FILE, "a") as f:
        for p in paths:
            f.write(p.rstrip(";").strip() + ";\n")


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def evaluate_candidate(source: str) -> Optional[List[Optional[int]]]:
    """Parse+length-check+evaluate. Returns output sequence or None if rejected."""
    if len(source) > MAX_SOURCE_LEN:
        return None
    raw = run_parser(source)
    if raw is None:
        return None
    seq = parse_output(raw)
    if not (MIN_OUTPUT_LEN <= len(seq) <= MAX_OUTPUT_LEN):
        return None
    # Reject all-rest outputs — they carry no behavioral signal.
    if all(v is None for v in seq):
        return None
    return seq


def make_candidate(archive: NoveltyArchive) -> Optional[str]:
    """Sample a generation strategy and produce one candidate source string."""
    strategy = random.choices(
        ["random", "mutate", "crossover"],
        weights=[1, 3, 2] if len(archive.entries) >= 2 else [1, 0, 0],
        k=1,
    )[0]

    if strategy == "random" or not archive.entries:
        tree = gen_tree(MAX_AST_DEPTH)
    elif strategy == "mutate":
        parent = random.choice(archive.entries).source
        tree = _try_parse_back(parent) or gen_tree(MAX_AST_DEPTH)
        tree = mutate(tree, rate=random.choice([0.15, 0.25, 0.4]))
    else:
        a, b = random.sample(archive.entries, 2)
        ta = _try_parse_back(a.source) or gen_tree(MAX_AST_DEPTH)
        tb = _try_parse_back(b.source) or gen_tree(MAX_AST_DEPTH)
        tree = crossover(ta, tb)
        if random.random() < 0.5:
            tree = mutate(tree, rate=0.2)

    if ast_size(tree) > MAX_AST_NODES or ast_depth(tree) > MAX_AST_DEPTH:
        return None
    return ast_render(tree)


def _try_parse_back(source: str) -> Optional[Node]:
    """Tokenize/parse an alien DSL string back into our Node structure."""
    try:
        tokens = _tokenize(source)
        pos = [0]
        node = _parse_expr(tokens, pos)
        if pos[0] != len(tokens):
            return None
        return node
    except Exception:
        return None


def _tokenize(s: str) -> List[str]:
    toks: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "()":
            toks.append(c)
            i += 1
        elif c == "-" and (i + 1 >= len(s) or not s[i + 1].isdigit()):
            toks.append("-")
            i += 1
        elif c.isdigit() or (c == "-" and i + 1 < len(s) and s[i + 1].isdigit()):
            j = i
            if s[i] == "-":
                i += 1
            while i < len(s) and s[i].isdigit():
                i += 1
            toks.append(s[j:i])
        elif c.isalpha() or c == "_":
            j = i
            while i < len(s) and (s[i].isalnum() or s[i] == "_"):
                i += 1
            toks.append(s[j:i])
        else:
            raise ValueError(f"bad char: {c}")
    return toks


def _parse_expr(tokens: List[str], pos: List[int]) -> Node:
    tok = tokens[pos[0]]
    if tok == "(":
        pos[0] += 1
        op = tokens[pos[0]]
        pos[0] += 1
        children: List[Node] = []
        while tokens[pos[0]] != ")":
            children.append(_parse_expr(tokens, pos))
        pos[0] += 1  # consume ')'
        if op not in OPERATORS:
            raise ValueError(f"unknown op: {op}")
        return Node(kind=op, children=children)
    if tok == "-":
        pos[0] += 1
        return Node(kind="rest")
    pos[0] += 1
    return Node(kind="num", value=int(tok))


def run(iterations: int = 2000, verbose_every: int = 100) -> None:
    if not os.path.exists(ALIEN_PARSER):
        raise SystemExit(f"alien_parser not found at {ALIEN_PARSER}. Run `make` first.")

    archive = NoveltyArchive()

    # Seed the archive. Seeds bypass the novelty threshold so the search has
    # a starting manifold to drift away from.
    for s in SEEDS:
        seq = evaluate_candidate(s)
        if seq is None:
            continue
        feat = embed(seq)
        archive.entries.append(ArchiveEntry(s, feat, 0.0, time.time()))
        archive._seen.add(s)

    accepted: List[str] = []
    attempts = 0
    parser_rejects = 0
    dup_rejects = 0

    print(f"alien novelty search — {iterations} iterations")
    print(f"  archive seeded with {len(archive.entries)} patterns")
    print(f"  k={archive.k}, initial threshold={archive.threshold:.3f}")
    print(f"  constraints: source<={MAX_SOURCE_LEN}, nodes<={MAX_AST_NODES}, "
          f"output in [{MIN_OUTPUT_LEN},{MAX_OUTPUT_LEN}]")
    print()

    start = time.time()
    for i in range(iterations):
        attempts += 1
        source = make_candidate(archive)
        if source is None:
            parser_rejects += 1
            continue
        seq = evaluate_candidate(source)
        if seq is None:
            parser_rejects += 1
            continue
        feat = embed(seq)
        added, nov = archive.consider(source, feat)
        if added:
            accepted.append(source)
        else:
            if nov == 0.0:
                dup_rejects += 1

        if verbose_every and (i + 1) % verbose_every == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[{i+1:5d}] archive={len(archive.entries):4d} "
                  f"threshold={archive.threshold:.3f} "
                  f"accepted={len(accepted):4d} "
                  f"parser_rej={parser_rejects:4d} "
                  f"dup={dup_rejects:4d} "
                  f"({rate:.1f} iters/s)")

    # Persist the full archive (minus seeds, which the user already has).
    novel_sources = [e.source for e in archive.entries if e.source not in SEEDS]
    append_patterns(novel_sources)

    print()
    print(f"done. {len(novel_sources)} novel patterns appended to {PATTERNS_FILE}")
    print(f"final threshold: {archive.threshold:.3f}")
    print(f"archive size: {len(archive.entries)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Novelty search for Alien DSL patterns.")
    ap.add_argument("--iters", type=int, default=2000, help="search iterations")
    ap.add_argument("--verbose-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    run(iterations=args.iters, verbose_every=args.verbose_every)
