#!/usr/bin/env python3
"""
alien_evolve.py - N-gram genetic evolution of alien DSL patterns

A Python prototype that generates increasingly interesting alien DSL patterns
using a combination of:
1. N-gram language model for argument generation
2. Genetic algorithms for structure evolution  
3. Rolling corpus that improves over time
4. External validation using alien_parser CLI

All generated patterns are validated before output.
"""

import random
import subprocess
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import time

# Path to alien_parser CLI
ALIEN_PARSER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'alien_parser')

# Path to persistent patterns file
PATTERNS_FILE = os.path.join(os.path.dirname(__file__), 'patterns.txt')

def load_patterns_file() -> List[str]:
    """Load patterns from patterns.txt if it exists."""
    if not os.path.exists(PATTERNS_FILE):
        return []

    patterns = []
    try:
        with open(PATTERNS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Strip trailing semicolon if present
                if line.endswith(';'):
                    line = line[:-1].strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    except Exception as e:
        print(f"Warning: Could not load {PATTERNS_FILE}: {e}")

    return patterns

def save_patterns_file(patterns: List[str], append: bool = True):
    """Save patterns to patterns.txt with semicolons and newlines."""
    mode = 'a' if append else 'w'
    try:
        with open(PATTERNS_FILE, mode) as f:
            for pattern in patterns:
                # Strip any existing semicolon to avoid duplicates
                pattern = pattern.rstrip(';').strip()
                f.write(pattern + ';\n')
    except Exception as e:
        print(f"Warning: Could not save to {PATTERNS_FILE}: {e}")

def filter_by_diversity(patterns: List[str], max_patterns: int = 30, 
                        similarity_threshold: float = 0.6) -> List[str]:
    """
    Filter patterns to maximize diversity.
    Uses greedy selection: pick highest fitness, then pick next that's 
    most different from already selected.
    """
    if len(patterns) <= max_patterns:
        return patterns
    
    # Parse all patterns and compute fingerprints
    parsed = []
    for p in patterns:
        ast = parse(p)
        if ast:
            parsed.append((p, ast, get_structural_fingerprint(ast)))
    
    if not parsed:
        return patterns[:max_patterns]
    
    # Greedy diverse selection
    selected = [parsed[0]]  # Start with first (highest fitness)
    remaining = parsed[1:]
    
    while len(selected) < max_patterns and remaining:
        # Find pattern most different from all selected
        best_idx = 0
        best_min_dist = 0
        
        for i, (p, ast, fp) in enumerate(remaining):
            # Compute minimum distance to any selected pattern
            min_sim = 1.0
            for sel_p, sel_ast, sel_fp in selected:
                sim = structural_similarity(ast, sel_ast)
                min_sim = min(min_sim, sim)
            
            # We want the one with lowest max similarity (most different)
            if (1 - min_sim) > best_min_dist:
                best_min_dist = 1 - min_sim
                best_idx = i
        
        # Check if it's different enough
        _, best_ast, _ = remaining[best_idx]
        too_similar = False
        for _, sel_ast, _ in selected:
            if structural_similarity(best_ast, sel_ast) > similarity_threshold:
                too_similar = True
                break
        
        if not too_similar:
            selected.append(remaining[best_idx])
        
        remaining.pop(best_idx)
        
        # If we've exhausted remaining, break
        if not remaining:
            break
    
    return [p for p, _, _ in selected]

def ensure_rest_diversity(patterns: List[str]) -> List[str]:
    """Ensure patterns have good rest distribution - filter out extremes."""
    result = []
    for p in patterns:
        ast = parse(p)
        if ast:
            numbers, hyphens = count_leaves(ast)
            total = numbers + hyphens
            if total > 0 and numbers > 0:  # Must have at least one note
                ratio = hyphens / total
                # Accept patterns with 10-80% rests
                if 0.1 <= ratio <= 0.8:
                    result.append(p)
    return result

# =============================================================================
# DSL SPECIFICATION (from alien_core.h analysis)
# =============================================================================

# Operator specifications: (min_args, max_args, arg_types)
# arg_types: 'any' = any expression, 'num' = single number, 'seq' = sequence
# Based on C parser eval functions
OPERATORS = {
    # Core - flexible args
    'seq': (1, None, ['any']),  # any number of any args
    'rep': (2, None, ['any', 'num']),  # patterns..., count (last must be single num)
    
    # Arithmetic - pattern + single number
    'add': (2, 2, ['any', 'num']),
    'mul': (2, 2, ['any', 'num']),
    'mod': (2, 2, ['any', 'num']),  # divisor must be positive
    'scale': (5, 5, ['any', 'num', 'num', 'num', 'num']),
    'clamp': (3, 3, ['any', 'num', 'num']),
    
    # Rhythm generators
    'euclid': (2, 3, ['any', 'num', 'num']),  # pattern/hits, steps, [rotation]
    'bjork': (2, 2, ['num', 'num']),  # hits, steps (both single numbers)
    'subdiv': (2, 2, ['any', 'num']),  # pattern, subdivisions
    
    # List manipulation - single pattern arg
    'reverse': (1, 1, ['any']),
    'rotate': (2, 2, ['any', 'num']),
    'palindrome': (1, 1, ['any']),
    'mirror': (1, 1, ['any']),
    'interleave': (2, 2, ['any', 'any']),
    'shuffle': (1, 1, ['any']),
    
    # Selection
    'take': (2, 2, ['any', 'num']),  # n must be non-negative
    'drop': (2, 2, ['any', 'num']),  # n must be non-negative
    'every': (2, 2, ['any', 'num']),  # n must be positive
    'slice': (3, 3, ['any', 'num', 'num']),
    'filter': (1, 1, ['any']),
    
    # Randomness
    'choose': (2, None, ['any']),  # at least 2 choices
    'rand': (3, 3, ['num', 'num', 'num']),  # count, min, max
    'prob': (2, 2, ['any', 'num']),  # pattern, probability 0-100
    'maybe': (3, 3, ['any', 'any', 'num']),  # seq1, seq2, probability
    'degrade': (2, 2, ['any', 'num']),  # pattern, probability 0-100
    
    # Pattern generation - all single numbers
    'range': (2, 3, ['num', 'num', 'num']),  # start, end, [step]
    'ramp': (3, 3, ['num', 'num', 'num']),  # start, end, steps
    'drunk': (3, 3, ['num', 'num', 'num']),  # steps, max_step, start
    
    # Conditional
    'cycle': (2, 2, ['any', 'num']),  # pattern, length (positive)
    'grow': (1, 1, ['any']),
    
    # Musical
    'transpose': (2, 2, ['any', 'num']),  # same as add
    'quantize': (2, 2, ['any', 'any']),  # pattern, scale (scale must have values)
    'chord': (2, 2, ['num', 'num']),  # root, type (0-6)
    'arp': (3, 3, ['any', 'num', 'num']),  # pattern, direction (0-2), length
    
    # Time
    'delay': (2, 2, ['any', 'num']),  # pattern, n (non-negative)
    'gate': (2, 2, ['any', 'num']),  # pattern, n (positive)
}

# =============================================================================
# EXTERNAL VALIDATION
# =============================================================================

def validate_with_parser(program: str) -> bool:
    """Validate program using the alien_parser CLI."""
    if not os.path.exists(ALIEN_PARSER_PATH):
        return validate(program)  # fallback to internal validation

    try:
        result = subprocess.run(
            [ALIEN_PARSER_PATH, program],
            capture_output=True,
            text=True,
            timeout=2
        )
        # If it exits 0 and produces output (not error), it's valid
        return result.returncode == 0 and not result.stderr
    except (subprocess.TimeoutExpired, Exception):
        return False

def evaluate_pattern_output(program: str) -> Optional[str]:
    """Get the evaluated output from alien_parser CLI."""
    if not os.path.exists(ALIEN_PARSER_PATH):
        return None

    try:
        result = subprocess.run(
            [ALIEN_PARSER_PATH, program],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None

def output_rest_ratio(program: str) -> float:
    """
    Calculate the ratio of rests in the evaluated output.
    Returns 1.0 if all rests, 0.0 if no rests, -1.0 if evaluation failed.
    """
    output = evaluate_pattern_output(program)
    if output is None:
        return -1.0

    # Parse the output - expect space-separated values where "-" or "-1" is a rest
    values = output.split()
    if not values:
        return 1.0  # Empty output = all rests

    rest_count = 0
    note_count = 0
    for v in values:
        v = v.strip()
        if v == '-' or v == '-1':
            rest_count += 1
        elif v.lstrip('-').isdigit():
            note_count += 1

    total = rest_count + note_count
    if total == 0:
        return 1.0  # No valid values = treat as all rests

    return rest_count / total

# =============================================================================
# INTERNAL PARSER & VALIDATOR
# =============================================================================

class ParseError(Exception):
    pass

def tokenize(s: str) -> List[str]:
    """Tokenize alien DSL string into list of tokens."""
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] == '(':
            tokens.append('(')
            i += 1
        elif s[i] == ')':
            tokens.append(')')
            i += 1
        elif s[i] == '-' and (i + 1 >= len(s) or not s[i+1].isdigit()):
            tokens.append('-')
            i += 1
        elif s[i].isdigit() or (s[i] == '-' and i + 1 < len(s) and s[i+1].isdigit()):
            j = i
            if s[i] == '-':
                i += 1
            while i < len(s) and s[i].isdigit():
                i += 1
            tokens.append(s[j:i])
        elif s[i].isalpha():
            j = i
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                i += 1
            tokens.append(s[j:i])
        else:
            raise ParseError(f"Invalid character: {s[i]}")
    return tokens

@dataclass
class ASTNode:
    """AST node for alien DSL."""
    type: str  # 'number', 'hyphen', or operator name
    value: Optional[int] = None
    children: List['ASTNode'] = field(default_factory=list)

def parse(s: str) -> Optional[ASTNode]:
    """Parse alien DSL string into AST. Returns None if invalid."""
    try:
        tokens = tokenize(s)
        if not tokens:
            return None
        pos = [0]
        result = parse_expr(tokens, pos)
        if pos[0] != len(tokens):
            return None
        return result
    except (ParseError, IndexError, ValueError):
        return None

def parse_expr(tokens: List[str], pos: List[int]) -> ASTNode:
    """Parse a single expression."""
    if pos[0] >= len(tokens):
        raise ParseError("Unexpected end of input")
    
    tok = tokens[pos[0]]
    
    if tok == '(':
        pos[0] += 1
        if pos[0] >= len(tokens):
            raise ParseError("Expected operator")
        op = tokens[pos[0]]
        if op not in OPERATORS:
            raise ParseError(f"Unknown operator: {op}")
        pos[0] += 1
        
        children = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ')':
            children.append(parse_expr(tokens, pos))
        
        if pos[0] >= len(tokens) or tokens[pos[0]] != ')':
            raise ParseError("Expected ')'")
        pos[0] += 1
        
        return ASTNode(type=op, children=children)
    
    elif tok == '-':
        pos[0] += 1
        return ASTNode(type='hyphen')
    
    elif tok.lstrip('-').isdigit():
        pos[0] += 1
        return ASTNode(type='number', value=int(tok))
    
    else:
        raise ParseError(f"Unexpected token: {tok}")

def validate(s: str) -> bool:
    """Check if string is valid alien DSL (internal check)."""
    ast = parse(s)
    if ast is None:
        return False
    return validate_ast(ast)

def validate_ast(node: ASTNode, max_depth: int = 5, current_depth: int = 0) -> bool:
    """Validate AST structure based on C parser rules."""
    if current_depth > max_depth:
        return False
    
    if node.type == 'hyphen':
        return True
    
    if node.type == 'number':
        if node.value is None:
            return False
        return True
    
    if node.type not in OPERATORS:
        return False
    
    min_args, max_args, arg_types = OPERATORS[node.type]
    n_children = len(node.children)
    
    if n_children < min_args:
        return False
    if max_args is not None and n_children > max_args:
        return False
    
    # Specific semantic validation based on C evaluator
    if node.type == 'chord':
        if n_children < 2:
            return False
        # Both args must evaluate to single numbers
        root = node.children[0]
        typ = node.children[1]
        if root.type != 'number' or typ.type != 'number':
            return False
        if typ.value < 0 or typ.value > 6:
            return False
    
    if node.type == 'arp':
        if n_children < 3:
            return False
        # direction and length must be single numbers
        direction = node.children[1]
        length = node.children[2]
        if direction.type != 'number' or length.type != 'number':
            return False
        if direction.value < 0 or direction.value > 2:
            return False
        if length.value <= 0:
            return False
    
    if node.type == 'bjork':
        if n_children < 2:
            return False
        hits = node.children[0]
        steps = node.children[1]
        if hits.type != 'number' or steps.type != 'number':
            return False
        if steps.value <= 0 or steps.value > 256:
            return False
    
    if node.type == 'range':
        # All args must be single numbers
        for child in node.children:
            if child.type != 'number':
                return False
        if n_children == 3 and node.children[2].value == 0:
            return False  # step cannot be 0
    
    if node.type == 'ramp':
        for child in node.children:
            if child.type != 'number':
                return False
    
    if node.type == 'drunk':
        for child in node.children:
            if child.type != 'number':
                return False
    
    if node.type == 'rand':
        for child in node.children:
            if child.type != 'number':
                return False
    
    if node.type in ('add', 'mul', 'transpose'):
        if n_children >= 2:
            delta = node.children[1]
            if delta.type != 'number':
                return False
    
    if node.type == 'mod':
        if n_children >= 2:
            divisor = node.children[1]
            if divisor.type != 'number' or divisor.value <= 0:
                return False
    
    if node.type in ('take', 'drop', 'delay'):
        if n_children >= 2:
            n = node.children[1]
            if n.type != 'number' or n.value < 0:
                return False
    
    if node.type in ('every', 'gate', 'cycle', 'subdiv'):
        if n_children >= 2:
            n = node.children[1]
            if n.type != 'number' or n.value <= 0:
                return False
    
    if node.type == 'euclid':
        if n_children >= 2:
            steps = node.children[1]
            if steps.type != 'number' or steps.value <= 0:
                return False
    
    if node.type in ('prob', 'degrade', 'maybe'):
        # probability arg must be number
        prob_idx = 1 if node.type in ('prob', 'degrade') else 2
        if n_children > prob_idx:
            prob = node.children[prob_idx]
            if prob.type != 'number':
                return False
    
    if node.type == 'scale':
        # All 4 range args must be numbers
        for i in range(1, min(5, n_children)):
            if node.children[i].type != 'number':
                return False
    
    if node.type == 'clamp':
        if n_children >= 3:
            if node.children[1].type != 'number' or node.children[2].type != 'number':
                return False
    
    if node.type == 'slice':
        if n_children >= 3:
            if node.children[1].type != 'number' or node.children[2].type != 'number':
                return False
    
    if node.type == 'rotate':
        if n_children >= 2:
            if node.children[1].type != 'number':
                return False
    
    return all(validate_ast(child, max_depth, current_depth + 1) for child in node.children)

def ast_to_string(node: ASTNode) -> str:
    """Convert AST back to string."""
    if node.type == 'number':
        return str(node.value)
    elif node.type == 'hyphen':
        return '-'
    else:
        children_str = ' '.join(ast_to_string(c) for c in node.children)
        return f"({node.type} {children_str})"

# =============================================================================
# N-GRAM MODEL
# =============================================================================

class NgramModel:
    """Character-level n-gram model with backoff."""
    
    def __init__(self, max_n: int = 10):
        self.max_n = max_n
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.totals: Dict[str, int] = defaultdict(int)
    
    def train(self, corpus: List[str]):
        """Train on list of programs."""
        for program in corpus:
            text = program + '\n'
            for n in range(1, self.max_n + 1):
                for i in range(len(text) - n):
                    context = text[i:i+n-1] if n > 1 else ''
                    char = text[i+n-1]
                    self.counts[context][char] += 1
                    self.totals[context] += 1
    
    def sample_next(self, context: str, temperature: float = 1.0) -> Optional[str]:
        """Sample next character given context."""
        for n in range(min(len(context), self.max_n - 1), -1, -1):
            ctx = context[-n:] if n > 0 else ''
            if ctx in self.counts and self.totals[ctx] > 0:
                total = self.totals[ctx]
                dist = {ch: count / total for ch, count in self.counts[ctx].items()}
                
                if temperature != 1.0:
                    items = list(dist.items())
                    weights = [w ** (1.0 / temperature) for _, w in items]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    return random.choices([ch for ch, _ in items], weights=weights)[0]
                
                chars, probs = zip(*dist.items())
                return random.choices(chars, weights=probs)[0]
        return None
    
    def complete(self, prefix: str, max_len: int = 100) -> str:
        """Complete a string until balanced parens or max length."""
        result = prefix
        paren_depth = prefix.count('(') - prefix.count(')')
        
        for _ in range(max_len):
            ch = self.sample_next(result)
            if ch is None or ch == '\n':
                break
            result += ch
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    break
        
        return result

# =============================================================================
# SMART STRUCTURE GENERATION
# =============================================================================

def random_number(context: str = None) -> int:
    """Generate a musically-meaningful number based on context."""
    if context == 'midi':
        return random.choice([48, 52, 55, 60, 62, 64, 65, 67, 69, 71, 72])
    elif context == 'steps':
        return random.choice([4, 8, 16])
    elif context == 'hits':
        return random.choice([2, 3, 4, 5, 7])
    elif context == 'chord_type':
        return random.choice([0, 1, 4, 5])
    elif context == 'direction':
        return random.choice([0, 1, 2])
    elif context == 'small':
        return random.choice([1, 2, 3, 4])
    elif context == 'probability':
        return random.choice([25, 50, 75])
    elif context == 'interval':
        return random.choice([-12, -7, -5, 5, 7, 12])
    else:
        choice = random.random()
        if choice < 0.4:
            return random.choice([48, 60, 64, 67, 72])
        elif choice < 0.6:
            return random.choice([4, 8, 16])
        elif choice < 0.8:
            return random.choice([2, 3, 4, 5])
        else:
            return random.choice([1, 2, 3])

def random_leaf(hyphen_prob: float = 0.25) -> ASTNode:
    """Generate a random leaf node with configurable hyphen probability."""
    if random.random() < hyphen_prob:
        return ASTNode(type='hyphen')
    return ASTNode(type='number', value=random_number())

def random_seq_with_hyphens(length: int = 4, hyphen_prob: float = 0.3) -> ASTNode:
    """Generate a seq with mixed numbers and hyphens."""
    children = []
    for _ in range(length):
        if random.random() < hyphen_prob:
            children.append(ASTNode(type='hyphen'))
        else:
            children.append(ASTNode(type='number', value=random_number('midi')))
    return ASTNode(type='seq', children=children)

def generate_for_operator(op: str, max_depth: int = 3, current_depth: int = 0) -> ASTNode:
    """Generate valid children for a specific operator based on C parser rules."""
    
    def gen_pattern(depth):
        """Generate a pattern (can be nested or leaf)."""
        if depth >= max_depth or random.random() < 0.4:
            return random_leaf(hyphen_prob=0.2)
        return random_structure(max_depth, depth)
    
    def gen_seq_pattern(depth):
        """Generate a sequence pattern with hyphens."""
        if depth >= max_depth or random.random() < 0.5:
            return random_seq_with_hyphens(random.randint(3, 6), hyphen_prob=0.3)
        return random_structure(max_depth, depth)
    
    if op == 'seq':
        n = random.randint(2, 5)
        children = []
        for _ in range(n):
            if random.random() < 0.3:
                children.append(ASTNode(type='hyphen'))
            elif random.random() < 0.5:
                children.append(ASTNode(type='number', value=random_number('midi')))
            else:
                children.append(gen_pattern(current_depth + 1))
        return ASTNode(type='seq', children=children)
    
    elif op == 'euclid':
        # (euclid pattern/hits steps [rotation])
        if random.random() < 0.6:
            # Pattern mode - distribute notes
            pattern = gen_seq_pattern(current_depth + 1)
            steps = ASTNode(type='number', value=random_number('steps'))
            children = [pattern, steps]
        else:
            # Hits mode
            hits = ASTNode(type='number', value=random_number('hits'))
            steps = ASTNode(type='number', value=random_number('steps'))
            children = [hits, steps]
        if random.random() < 0.2:
            children.append(ASTNode(type='number', value=random.randint(0, 3)))
        return ASTNode(type='euclid', children=children)
    
    elif op == 'bjork':
        hits = ASTNode(type='number', value=random_number('hits'))
        steps = ASTNode(type='number', value=random_number('steps'))
        return ASTNode(type='bjork', children=[hits, steps])
    
    elif op == 'chord':
        root = ASTNode(type='number', value=random_number('midi'))
        typ = ASTNode(type='number', value=random_number('chord_type'))
        return ASTNode(type='chord', children=[root, typ])
    
    elif op == 'arp':
        # (arp pattern direction length)
        pattern = generate_for_operator('chord', max_depth, current_depth + 1)
        direction = ASTNode(type='number', value=random_number('direction'))
        length = ASTNode(type='number', value=random_number('steps'))
        return ASTNode(type='arp', children=[pattern, direction, length])
    
    elif op == 'range':
        start = random_number('midi')
        end = start + random.choice([4, 7, 12])
        children = [
            ASTNode(type='number', value=start),
            ASTNode(type='number', value=end)
        ]
        if random.random() < 0.2:
            children.append(ASTNode(type='number', value=random.choice([1, 2])))
        return ASTNode(type='range', children=children)
    
    elif op == 'ramp':
        start = random_number('midi')
        end = start + random.choice([7, 12, 24])
        steps = random_number('steps')
        return ASTNode(type='ramp', children=[
            ASTNode(type='number', value=start),
            ASTNode(type='number', value=end),
            ASTNode(type='number', value=steps)
        ])
    
    elif op == 'drunk':
        steps = random_number('steps')
        max_step = random.choice([1, 2, 3])
        start = random_number('midi')
        return ASTNode(type='drunk', children=[
            ASTNode(type='number', value=steps),
            ASTNode(type='number', value=max_step),
            ASTNode(type='number', value=start)
        ])
    
    elif op == 'rand':
        count = random_number('steps')
        min_val = random.choice([0, 1, 60])
        max_val = min_val + random.choice([1, 6, 12])
        return ASTNode(type='rand', children=[
            ASTNode(type='number', value=count),
            ASTNode(type='number', value=min_val),
            ASTNode(type='number', value=max_val)
        ])
    
    elif op in ('reverse', 'palindrome', 'mirror', 'shuffle', 'filter', 'grow'):
        return ASTNode(type=op, children=[gen_seq_pattern(current_depth + 1)])
    
    elif op in ('rotate', 'take', 'drop', 'every', 'gate', 'delay', 'cycle', 'subdiv'):
        pattern = gen_seq_pattern(current_depth + 1)
        n = ASTNode(type='number', value=random_number('small'))
        return ASTNode(type=op, children=[pattern, n])
    
    elif op == 'slice':
        pattern = gen_seq_pattern(current_depth + 1)
        start = random.randint(0, 2)
        end = start + random.randint(2, 4)
        return ASTNode(type='slice', children=[
            pattern,
            ASTNode(type='number', value=start),
            ASTNode(type='number', value=end)
        ])
    
    elif op == 'interleave':
        p1 = gen_pattern(current_depth + 1)
        p2 = gen_pattern(current_depth + 1)
        return ASTNode(type='interleave', children=[p1, p2])
    
    elif op == 'choose':
        n = random.randint(2, 3)
        children = [gen_pattern(current_depth + 1) for _ in range(n)]
        return ASTNode(type='choose', children=children)
    
    elif op in ('prob', 'degrade'):
        pattern = gen_seq_pattern(current_depth + 1)
        prob = ASTNode(type='number', value=random_number('probability'))
        return ASTNode(type=op, children=[pattern, prob])
    
    elif op == 'maybe':
        p1 = gen_pattern(current_depth + 1)
        p2 = gen_pattern(current_depth + 1)
        prob = ASTNode(type='number', value=random_number('probability'))
        return ASTNode(type='maybe', children=[p1, p2, prob])
    
    elif op in ('add', 'mul', 'transpose'):
        pattern = gen_pattern(current_depth + 1)
        delta = ASTNode(type='number', value=random_number('interval') if op == 'transpose' else random.choice([1, 2, 12]))
        return ASTNode(type=op, children=[pattern, delta])
    
    elif op == 'mod':
        pattern = gen_pattern(current_depth + 1)
        divisor = ASTNode(type='number', value=random.choice([2, 4, 7, 12]))
        return ASTNode(type='mod', children=[pattern, divisor])
    
    elif op == 'quantize':
        pattern = gen_pattern(current_depth + 1)
        # Scale as a seq of MIDI notes
        scale = ASTNode(type='seq', children=[
            ASTNode(type='number', value=v) for v in [60, 62, 64, 65, 67, 69, 71, 72]
        ])
        return ASTNode(type='quantize', children=[pattern, scale])
    
    elif op == 'scale':
        pattern = gen_pattern(current_depth + 1)
        return ASTNode(type='scale', children=[
            pattern,
            ASTNode(type='number', value=0),
            ASTNode(type='number', value=127),
            ASTNode(type='number', value=60),
            ASTNode(type='number', value=72)
        ])
    
    elif op == 'clamp':
        pattern = gen_pattern(current_depth + 1)
        return ASTNode(type='clamp', children=[
            pattern,
            ASTNode(type='number', value=48),
            ASTNode(type='number', value=84)
        ])
    
    elif op == 'rep':
        pattern = gen_pattern(current_depth + 1)
        count = ASTNode(type='number', value=random.choice([2, 3, 4]))
        return ASTNode(type='rep', children=[pattern, count])
    
    else:
        # Fallback
        return ASTNode(type='seq', children=[random_leaf(), random_leaf()])

# Operator categories for weighted selection - ALL operators covered
RHYTHM_OPS = ['euclid', 'bjork', 'gate', 'subdiv']
MELODY_OPS = ['chord', 'arp', 'range', 'quantize', 'transpose', 'ramp']
TRANSFORM_OPS = ['reverse', 'palindrome', 'rotate', 'interleave', 'mirror', 'shuffle']
STRUCTURE_OPS = ['seq', 'cycle', 'rep', 'grow']
RANDOM_OPS = ['prob', 'degrade', 'maybe', 'choose', 'drunk', 'rand']
SELECTION_OPS = ['take', 'drop', 'every', 'slice', 'filter']
ARITHMETIC_OPS = ['add', 'mul', 'mod', 'clamp', 'scale']
TIME_OPS = ['delay', 'gate']

# All operators that have generate_for_operator support
ALL_SUPPORTED_OPS = list(set(
    RHYTHM_OPS + MELODY_OPS + TRANSFORM_OPS + STRUCTURE_OPS + 
    RANDOM_OPS + SELECTION_OPS + TIME_OPS
))

def random_structure(max_depth: int = 3, current_depth: int = 0) -> ASTNode:
    """Generate a random valid AST structure using all available operators."""
    if current_depth >= max_depth:
        return random_leaf(hyphen_prob=0.25)
    
    # Weight operator selection by depth
    if current_depth == 0:
        # Top level: prefer musical/structural operators but include all
        weights = {
            'rhythm': 0.25,
            'melody': 0.25,
            'structure': 0.20,
            'transform': 0.15,
            'random': 0.10,
            'selection': 0.05,
        }
        r = random.random()
        if r < 0.25:
            ops = RHYTHM_OPS
        elif r < 0.50:
            ops = MELODY_OPS
        elif r < 0.70:
            ops = STRUCTURE_OPS
        elif r < 0.85:
            ops = TRANSFORM_OPS
        elif r < 0.95:
            ops = RANDOM_OPS
        else:
            ops = SELECTION_OPS + TIME_OPS
    else:
        # Nested: use all operators with slight preference for transforms
        r = random.random()
        if r < 0.30:
            ops = TRANSFORM_OPS
        elif r < 0.50:
            ops = ['seq', 'range', 'chord']
        elif r < 0.70:
            ops = RANDOM_OPS
        elif r < 0.85:
            ops = RHYTHM_OPS
        else:
            ops = SELECTION_OPS + MELODY_OPS
    
    op = random.choice(ops)
    return generate_for_operator(op, max_depth, current_depth)

# =============================================================================
# GENETIC OPERATORS - ENHANCED EVOLUTION
# =============================================================================
#
# Evolution Sophistication Analysis:
# ----------------------------------
# Current implementation uses several genetic programming techniques:
#
# 1. MUTATION TYPES:
#    - Point mutation: tweak numeric values by musical intervals
#    - Structural mutation: swap operators, add/remove children
#    - Rest insertion: convert notes to hyphens and vice versa
#    - Subtree replacement: replace entire subtrees with new random ones
#    - Operator promotion: wrap a node in a new operator
#
# 2. CROSSOVER TYPES:
#    - Subtree crossover: swap compatible subtrees between parents
#    - Uniform crossover: mix children from both parents
#    - Headless chicken: cross with randomly generated tree
#
# 3. SELECTION:
#    - Tournament selection with fitness-proportionate bias
#    - Elitism: protect high-fitness patterns from eviction
#    - Novelty pressure: reward patterns different from corpus
#
# 4. DIVERSITY MAINTENANCE:
#    - Rolling corpus with diversity-aware eviction
#    - Novelty scoring in fitness function
#    - Multiple generation methods (n-gram, mutation, crossover, random)
#
# Potential Improvements:
# - Island model: run parallel populations with migration
# - Coevolution: evolve rhythm and melody separately then combine
# - Semantic similarity: use output sequences for novelty, not just syntax
# - Adaptive mutation rates based on fitness stagnation
# - Grammar-guided crossover to ensure validity
# =============================================================================

# Compatible operator groups for smart crossover
OPERATOR_GROUPS = {
    'rhythm': {'euclid', 'bjork', 'gate', 'prob', 'degrade'},
    'melody': {'chord', 'arp', 'range', 'ramp', 'drunk', 'quantize', 'transpose'},
    'transform': {'reverse', 'palindrome', 'mirror', 'rotate', 'shuffle', 'interleave'},
    'structure': {'seq', 'rep', 'cycle', 'take', 'drop', 'slice', 'every', 'filter'},
    'random': {'choose', 'maybe', 'rand', 'prob', 'degrade'},
}

def get_operator_group(op: str) -> str:
    """Get the functional group of an operator."""
    for group, ops in OPERATOR_GROUPS.items():
        if op in ops:
            return group
    return 'structure'

def mutate_ast(node: ASTNode, mutation_rate: float = 0.15, depth: int = 0) -> ASTNode:
    """
    Enhanced AST mutation with multiple mutation types.
    
    Mutation types:
    - tweak: adjust numeric values by musical intervals
    - swap_hyphen: convert between notes and rests
    - replace_subtree: replace with new random subtree
    - wrap_operator: wrap node in a new operator
    - modify_operator: change operator to similar one
    - add_child: add a new child to operator
    - remove_child: remove a child from operator
    """
    # Decrease mutation rate with depth to preserve structure
    effective_rate = mutation_rate / (1 + depth * 0.3)
    
    if random.random() > effective_rate:
        if node.type in ('number', 'hyphen'):
            return ASTNode(type=node.type, value=node.value)
        return ASTNode(
            type=node.type,
            children=[mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
        )
    
    # Choose mutation type based on node type
    if node.type == 'number':
        mutation_type = random.choices(
            ['tweak', 'replace', 'swap_hyphen', 'wrap'],
            weights=[0.4, 0.2, 0.3, 0.1]
        )[0]
        
        if mutation_type == 'tweak':
            # Musical interval adjustments
            delta = random.choice([-12, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 12])
            return ASTNode(type='number', value=node.value + delta)
        elif mutation_type == 'swap_hyphen':
            return ASTNode(type='hyphen')
        elif mutation_type == 'wrap':
            # Wrap in a simple operator
            wrapper = random.choice(['reverse', 'palindrome', 'seq'])
            if wrapper == 'seq':
                return ASTNode(type='seq', children=[
                    ASTNode(type='number', value=node.value),
                    ASTNode(type='hyphen'),
                ])
            return ASTNode(type=wrapper, children=[
                ASTNode(type='number', value=node.value)
            ])
        else:
            return ASTNode(type='number', value=random_number())
    
    elif node.type == 'hyphen':
        mutation_type = random.choices(
            ['swap_to_note', 'keep', 'wrap'],
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        if mutation_type == 'swap_to_note':
            return ASTNode(type='number', value=random_number('midi'))
        elif mutation_type == 'wrap':
            return ASTNode(type='seq', children=[
                ASTNode(type='hyphen'),
                ASTNode(type='hyphen'),
            ])
        return ASTNode(type='hyphen')
    
    else:
        # Operator node - more sophisticated mutations
        mutation_type = random.choices(
            ['mutate_children', 'modify_operator', 'add_child', 'remove_child', 
             'replace_subtree', 'wrap_operator'],
            weights=[0.35, 0.15, 0.15, 0.1, 0.15, 0.1]
        )[0]
        
        if mutation_type == 'mutate_children':
            return ASTNode(
                type=node.type,
                children=[mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
            )
        
        elif mutation_type == 'modify_operator':
            # Change to similar operator in same group
            group = get_operator_group(node.type)
            candidates = list(OPERATOR_GROUPS.get(group, {'seq'}))
            if node.type in candidates:
                candidates.remove(node.type)
            if candidates:
                new_op = random.choice(candidates)
                # Try to generate valid children for new operator
                try:
                    new_node = generate_for_operator(new_op, max_depth=2, current_depth=depth)
                    return new_node
                except:
                    pass
            return ASTNode(
                type=node.type,
                children=[mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
            )
        
        elif mutation_type == 'add_child':
            min_args, max_args, _ = OPERATORS.get(node.type, (1, None, []))
            if max_args is None or len(node.children) < max_args:
                new_children = [mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
                new_children.append(random_leaf(hyphen_prob=0.3))
                return ASTNode(type=node.type, children=new_children)
            return ASTNode(
                type=node.type,
                children=[mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
            )
        
        elif mutation_type == 'remove_child':
            min_args, _, _ = OPERATORS.get(node.type, (1, None, []))
            if len(node.children) > min_args:
                new_children = [mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
                idx = random.randrange(len(new_children))
                new_children.pop(idx)
                return ASTNode(type=node.type, children=new_children)
            return ASTNode(
                type=node.type,
                children=[mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
            )
        
        elif mutation_type == 'replace_subtree':
            # Replace entire subtree with new random one
            return random_structure(max_depth=2, current_depth=depth)
        
        elif mutation_type == 'wrap_operator':
            # Wrap this node in another operator
            wrapper = random.choice(['reverse', 'palindrome', 'prob', 'interleave'])
            if wrapper == 'prob':
                return ASTNode(type='prob', children=[
                    ASTNode(type=node.type, children=node.children),
                    ASTNode(type='number', value=random.choice([50, 75]))
                ])
            elif wrapper == 'interleave':
                return ASTNode(type='interleave', children=[
                    ASTNode(type=node.type, children=node.children),
                    random_seq_with_hyphens(3, 0.5)
                ])
            else:
                return ASTNode(type=wrapper, children=[
                    ASTNode(type=node.type, children=node.children)
                ])
        
        return ASTNode(
            type=node.type,
            children=[mutate_ast(c, mutation_rate, depth + 1) for c in node.children]
        )

def crossover_ast(parent1: ASTNode, parent2: ASTNode) -> ASTNode:
    """
    Enhanced crossover with multiple strategies.
    
    Strategies:
    - Subtree swap: replace subtree from parent1 with one from parent2
    - Uniform: for each child position, pick from either parent
    - Blend: combine numeric values from both parents
    """
    strategy = random.choices(
        ['subtree', 'uniform', 'blend'],
        weights=[0.5, 0.3, 0.2]
    )[0]
    
    if strategy == 'subtree':
        # Collect all subtrees from parent2
        subtrees2 = []
        def collect(node, depth=0):
            subtrees2.append((node, depth))
            if node.type not in ('number', 'hyphen'):
                for c in node.children:
                    collect(c, depth + 1)
        collect(parent2)
        
        # Replace random subtree in parent1 with one from parent2
        def replace_random(node, depth=0):
            # Higher chance to replace at similar depths
            for subtree, sub_depth in subtrees2:
                if abs(depth - sub_depth) <= 1 and random.random() < 0.12:
                    return deep_copy_ast(subtree)
            
            if node.type in ('number', 'hyphen'):
                return ASTNode(type=node.type, value=node.value)
            return ASTNode(
                type=node.type,
                children=[replace_random(c, depth + 1) for c in node.children]
            )
        
        return replace_random(parent1)
    
    elif strategy == 'uniform':
        # For operators, mix children from both parents
        if parent1.type in ('number', 'hyphen'):
            return ASTNode(type=parent1.type, value=parent1.value)
        
        if parent2.type in ('number', 'hyphen'):
            return deep_copy_ast(parent1)
        
        # If same operator, mix children
        if parent1.type == parent2.type:
            new_children = []
            max_len = max(len(parent1.children), len(parent2.children))
            for i in range(max_len):
                if i < len(parent1.children) and i < len(parent2.children):
                    chosen = random.choice([parent1.children[i], parent2.children[i]])
                    new_children.append(deep_copy_ast(chosen))
                elif i < len(parent1.children):
                    new_children.append(deep_copy_ast(parent1.children[i]))
                else:
                    new_children.append(deep_copy_ast(parent2.children[i]))
            
            # Ensure we have valid number of children
            min_args, max_args, _ = OPERATORS.get(parent1.type, (1, None, []))
            if len(new_children) < min_args:
                new_children.extend([random_leaf() for _ in range(min_args - len(new_children))])
            if max_args and len(new_children) > max_args:
                new_children = new_children[:max_args]
            
            return ASTNode(type=parent1.type, children=new_children)
        
        # Different operators - use subtree strategy
        return crossover_ast(parent1, parent2)
    
    else:  # blend
        # For numbers, blend values; for operators, recurse
        if parent1.type == 'number' and parent2.type == 'number':
            # Weighted average with some randomness
            weight = random.random()
            blended = int(parent1.value * weight + parent2.value * (1 - weight))
            return ASTNode(type='number', value=blended)
        
        if parent1.type in ('number', 'hyphen'):
            return ASTNode(type=parent1.type, value=parent1.value)
        
        return ASTNode(
            type=parent1.type,
            children=[
                crossover_ast(c, random.choice(collect_all_nodes(parent2)))
                for c in parent1.children
            ]
        )

def deep_copy_ast(node: ASTNode) -> ASTNode:
    """Create a deep copy of an AST node."""
    if node.type in ('number', 'hyphen'):
        return ASTNode(type=node.type, value=node.value)
    return ASTNode(
        type=node.type,
        children=[deep_copy_ast(c) for c in node.children]
    )

def collect_all_nodes(node: ASTNode) -> List[ASTNode]:
    """Collect all nodes in an AST."""
    nodes = [node]
    if node.type not in ('number', 'hyphen'):
        for c in node.children:
            nodes.extend(collect_all_nodes(c))
    return nodes

# =============================================================================
# FITNESS FUNCTIONS
# =============================================================================

def get_structural_fingerprint(node: ASTNode) -> str:
    """Get a structural fingerprint of an AST for diversity comparison."""
    if node.type == 'number':
        return 'N'
    elif node.type == 'hyphen':
        return '-'
    else:
        children_fp = ''.join(get_structural_fingerprint(c) for c in node.children)
        return f"({node.type[:3]}{children_fp})"

def get_operator_sequence(node: ASTNode) -> List[str]:
    """Get sequence of operators used (for diversity)."""
    ops = []
    def collect(n):
        if n.type not in ('number', 'hyphen'):
            ops.append(n.type)
            for c in n.children:
                collect(c)
    collect(node)
    return ops

def count_leaves(node: ASTNode) -> tuple:
    """Count numbers and hyphens in AST. Returns (numbers, hyphens)."""
    numbers = 0
    hyphens = 0
    def walk(n):
        nonlocal numbers, hyphens
        if n.type == 'number':
            numbers += 1
        elif n.type == 'hyphen':
            hyphens += 1
        else:
            for c in n.children:
                walk(c)
    walk(node)
    return numbers, hyphens

def complexity_score(node: ASTNode) -> float:
    """Score based on structural complexity."""
    def count_nodes(n):
        if n.type in ('number', 'hyphen'):
            return 1
        return 1 + sum(count_nodes(c) for c in n.children)
    
    def max_depth(n, d=0):
        if n.type in ('number', 'hyphen'):
            return d
        if not n.children:
            return d
        return max(max_depth(c, d + 1) for c in n.children)
    
    def unique_ops(n):
        ops = set()
        def collect(node):
            if node.type not in ('number', 'hyphen'):
                ops.add(node.type)
                for c in node.children:
                    collect(c)
        collect(n)
        return len(ops)
    
    nodes = count_nodes(node)
    depth = max_depth(node)
    ops = unique_ops(node)
    numbers, hyphens = count_leaves(node)
    total_leaves = numbers + hyphens
    
    # Hyphen ratio: ideal is 20-40% of leaves
    if total_leaves > 0:
        hyphen_ratio = hyphens / total_leaves
        if hyphen_ratio < 0.1:
            hyphen_score = 0.2  # penalize no rests
        elif hyphen_ratio < 0.2:
            hyphen_score = 0.6
        elif hyphen_ratio <= 0.4:
            hyphen_score = 1.0  # ideal range
        elif hyphen_ratio <= 0.6:
            hyphen_score = 0.7
        else:
            hyphen_score = 0.3  # too many rests
    else:
        hyphen_score = 0.0
    
    # Ideal ranges
    node_score = max(0, 1 - abs(nodes - 10) / 12)
    depth_score = max(0, 1 - abs(depth - 2) / 3)
    op_score = min(ops / 3, 1.0)
    
    return (node_score + depth_score + op_score + hyphen_score) / 4

def structural_similarity(ast1: ASTNode, ast2: ASTNode) -> float:
    """Compare structural similarity of two ASTs."""
    fp1 = get_structural_fingerprint(ast1)
    fp2 = get_structural_fingerprint(ast2)
    
    # Jaccard similarity on operator sequences
    ops1 = set(get_operator_sequence(ast1))
    ops2 = set(get_operator_sequence(ast2))
    
    if not ops1 and not ops2:
        return 1.0
    
    op_sim = len(ops1 & ops2) / len(ops1 | ops2) if (ops1 | ops2) else 0
    
    # Fingerprint prefix similarity
    common = sum(1 for a, b in zip(fp1, fp2) if a == b)
    fp_sim = common / max(len(fp1), len(fp2)) if max(len(fp1), len(fp2)) > 0 else 0
    
    return 0.5 * op_sim + 0.5 * fp_sim

def novelty_score(program: str, corpus: List[str], sample_size: int = 50) -> float:
    """Score based on structural difference from existing corpus."""
    if not corpus:
        return 1.0
    
    ast = parse(program)
    if ast is None:
        return 0.0
    
    sample = random.sample(corpus, min(sample_size, len(corpus)))
    
    max_sim = 0.0
    for other in sample:
        other_ast = parse(other)
        if other_ast is None:
            continue
        sim = structural_similarity(ast, other_ast)
        max_sim = max(max_sim, sim)
    
    return 1 - max_sim

def musical_score(node: ASTNode) -> float:
    """Score based on musical meaningfulness."""
    score = 0.0
    has_rhythm = False
    has_melody = False
    has_hyphens = False
    
    def evaluate(n):
        nonlocal score, has_rhythm, has_melody, has_hyphens
        if n.type == 'hyphen':
            has_hyphens = True
        elif n.type == 'euclid':
            score += 0.25
            has_rhythm = True
        elif n.type == 'bjork':
            score += 0.2
            has_rhythm = True
        elif n.type == 'chord':
            score += 0.25
            has_melody = True
        elif n.type == 'arp':
            score += 0.2
            has_melody = True
        elif n.type == 'range':
            score += 0.15
            has_melody = True
        elif n.type in ('quantize', 'transpose'):
            score += 0.15
            has_melody = True
        elif n.type in ('reverse', 'palindrome', 'interleave'):
            score += 0.1
        elif n.type in ('prob', 'degrade', 'maybe'):
            score += 0.1
        elif n.type == 'seq':
            score += 0.05
        
        if n.type not in ('number', 'hyphen'):
            for c in n.children:
                evaluate(c)
    
    evaluate(node)
    
    if has_rhythm and has_melody:
        score += 0.2
    if has_hyphens:
        score += 0.1
    
    return min(score, 1.0)

def length_score(program: str) -> float:
    """Score based on program length."""
    length = len(program)
    if length < 15:
        return 0.3
    elif length < 40:
        return 1.0
    elif length < 70:
        return 0.9
    elif length < 100:
        return 0.6
    else:
        return 0.3

def rest_score(node: ASTNode) -> float:
    """Score specifically for rest/hyphen usage."""
    numbers, hyphens = count_leaves(node)
    total = numbers + hyphens
    
    if total == 0:
        return 0.0
    
    # All rests = useless pattern
    if numbers == 0:
        return 0.0
    
    ratio = hyphens / total
    
    # Ideal: 20-35% rests
    if ratio < 0.1:
        return 0.1  # strongly penalize no rests
    elif ratio < 0.15:
        return 0.4
    elif ratio < 0.2:
        return 0.7
    elif ratio <= 0.35:
        return 1.0  # ideal
    elif ratio <= 0.5:
        return 0.6
    elif ratio < 0.8:
        return 0.2  # too many rests
    else:
        return 0.0  # almost all rests = useless

def output_score(program: str) -> float:
    """
    Score based on the actual evaluated output.
    Penalizes patterns that produce all rests or too many rests.
    """
    ratio = output_rest_ratio(program)

    if ratio < 0:
        # Evaluation failed - use neutral score
        return 0.5

    if ratio >= 1.0:
        # All rests = useless pattern
        return 0.0
    elif ratio >= 0.9:
        # Almost all rests
        return 0.1
    elif ratio >= 0.7:
        # Too many rests
        return 0.3
    elif ratio >= 0.5:
        # Borderline
        return 0.5
    elif ratio >= 0.2:
        # Good balance of rests
        return 1.0
    elif ratio >= 0.1:
        # Decent amount of rests
        return 0.8
    else:
        # Very few rests in output
        return 0.6


def fitness(program: str, corpus: List[str]) -> float:
    """Combined fitness score - prioritizes novelty, output quality, and rest usage."""
    ast = parse(program)
    if ast is None:
        return 0.0

    c_score = complexity_score(ast)
    n_score = novelty_score(program, corpus)
    m_score = musical_score(ast)
    l_score = length_score(program)
    r_score = rest_score(ast)
    o_score = output_score(program)

    # Weights: novelty 30%, output 20%, rests 15%, musicality 20%, complexity 10%, length 5%
    # Output score ensures we penalize patterns that evaluate to all rests
    return (0.10 * c_score + 0.30 * n_score + 0.20 * m_score +
            0.05 * l_score + 0.15 * r_score + 0.20 * o_score)

# =============================================================================
# ROLLING CORPUS
# =============================================================================

@dataclass
class CorpusEntry:
    program: str
    fitness: float
    generation: int
    timestamp: float

class RollingCorpus:
    """Self-improving corpus with diversity-aware eviction."""
    
    def __init__(self, max_size: int = 500, protected_seeds: List[str] = None):
        self.max_size = max_size
        self.entries: List[CorpusEntry] = []
        self.seen: Set[str] = set()
        self.protected = set(protected_seeds or [])
        
        for seed in (protected_seeds or []):
            if validate(seed):
                self.entries.append(CorpusEntry(
                    program=seed,
                    fitness=1.0,
                    generation=0,
                    timestamp=time.time()
                ))
                self.seen.add(self._normalize(seed))
    
    def _normalize(self, program: str) -> str:
        import re
        return re.sub(r'\s+', ' ', program.strip())
    
    def add(self, program: str, fitness_score: float, generation: int) -> bool:
        normalized = self._normalize(program)
        
        if normalized in self.seen:
            return False
        
        if not validate(program):
            return False
        
        entry = CorpusEntry(
            program=program,
            fitness=fitness_score,
            generation=generation,
            timestamp=time.time()
        )
        self.entries.append(entry)
        self.seen.add(normalized)
        
        while len(self.entries) > self.max_size:
            self._evict()
        
        return True
    
    def _evict(self):
        candidates = [
            (i, e) for i, e in enumerate(self.entries)
            if e.program not in self.protected
        ]
        
        if not candidates:
            return
        
        now = time.time()
        def value(entry):
            age = now - entry.timestamp
            recency_bonus = 0.1 * max(0, 1 - age / 3600)
            return entry.fitness + recency_bonus
        
        idx, entry = min(candidates, key=lambda x: value(x[1]))
        self.seen.discard(self._normalize(entry.program))
        self.entries.pop(idx)
    
    def sample(self, n: int) -> List[str]:
        if n >= len(self.entries):
            return [e.program for e in self.entries]
        return [e.program for e in random.sample(self.entries, n)]
    
    def get_programs(self) -> List[str]:
        return [e.program for e in self.entries]
    
    def stats(self) -> dict:
        if not self.entries:
            return {'size': 0, 'avg_fitness': 0}
        return {
            'size': len(self.entries),
            'avg_fitness': sum(e.fitness for e in self.entries) / len(self.entries),
        }

# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

class AlienEvolver:
    """Main evolution engine with external validation."""
    
    def __init__(self, seed_corpus: List[str], max_corpus_size: int = 500):
        self.corpus = RollingCorpus(max_size=max_corpus_size, protected_seeds=seed_corpus)
        self.ngram = NgramModel(max_n=12)
        self.generation = 0
        self.use_external_validation = os.path.exists(ALIEN_PARSER_PATH)
        
        self._rebuild_ngram()
    
    def _rebuild_ngram(self):
        programs = self.corpus.get_programs()
        if programs:
            self.ngram = NgramModel(max_n=12)
            self.ngram.train(programs)
    
    def _validate(self, program: str) -> bool:
        """Validate using external parser if available."""
        if self.use_external_validation:
            return validate_with_parser(program)
        return validate(program)
    
    def generate_with_ngram(self) -> Optional[str]:
        """Generate a program using n-gram completion."""
        starters = [
            '(seq ', '(euclid ', '(chord ', '(arp ', 
            '(interleave ', '(reverse ', '(palindrome ',
            '(bjork ', '(range ', '(prob ', '(degrade '
        ]
        start = random.choice(starters)
        result = self.ngram.complete(start, max_len=120)
        
        if self._validate(result):
            return result
        return None
    
    def generate_random(self) -> Optional[str]:
        """Generate a random valid program."""
        for _ in range(10):
            ast = random_structure(max_depth=3)
            program = ast_to_string(ast)
            if self._validate(program):
                return program
        return None
    
    def mutate_existing(self) -> Optional[str]:
        """Mutate an existing program."""
        programs = self.corpus.sample(1)
        if not programs:
            return None
        
        ast = parse(programs[0])
        if ast is None:
            return None
        
        for _ in range(10):
            mutated = mutate_ast(ast, mutation_rate=0.2)
            program = ast_to_string(mutated)
            if self._validate(program):
                return program
        return None
    
    def crossover_existing(self) -> Optional[str]:
        """Crossover two existing programs."""
        programs = self.corpus.sample(2)
        if len(programs) < 2:
            return None
        
        ast1 = parse(programs[0])
        ast2 = parse(programs[1])
        if ast1 is None or ast2 is None:
            return None
        
        for _ in range(10):
            child = crossover_ast(ast1, ast2)
            program = ast_to_string(child)
            if self._validate(program):
                return program
        return None
    
    def run_generation(self, candidates_per_gen: int = 50, 
                       min_fitness: float = 0.35,
                       rebuild_interval: int = 10) -> dict:
        """Run one generation of evolution with detailed rejection tracking."""
        self.generation += 1
        
        # Track attempts and rejections by method
        stats_by_method = {
            'ngram': {'attempts': 0, 'valid': 0, 'admitted': 0},
            'mutate': {'attempts': 0, 'valid': 0, 'admitted': 0},
            'crossover': {'attempts': 0, 'valid': 0, 'admitted': 0},
            'random': {'attempts': 0, 'valid': 0, 'admitted': 0},
        }
        
        candidates = []
        candidate_methods = []  # Track which method produced each candidate
        
        methods = {
            'ngram': 0.30,
            'mutate': 0.30,
            'crossover': 0.20,
            'random': 0.20,
        }
        
        for _ in range(candidates_per_gen):
            r = random.random()
            cumulative = 0
            method = 'random'
            
            for m, prob in methods.items():
                cumulative += prob
                if r < cumulative:
                    method = m
                    break
            
            stats_by_method[method]['attempts'] += 1
            
            candidate = None
            if method == 'ngram':
                candidate = self.generate_with_ngram()
            elif method == 'mutate':
                candidate = self.mutate_existing()
            elif method == 'crossover':
                candidate = self.crossover_existing()
            else:
                candidate = self.generate_random()
            
            if candidate:
                stats_by_method[method]['valid'] += 1
                candidates.append(candidate)
                candidate_methods.append(method)
        
        admitted = 0
        rejected_fitness = 0
        rejected_duplicate = 0
        corpus_programs = self.corpus.get_programs()
        
        for i, candidate in enumerate(candidates):
            f = fitness(candidate, corpus_programs)
            if f >= min_fitness:
                if self.corpus.add(candidate, f, self.generation):
                    admitted += 1
                    stats_by_method[candidate_methods[i]]['admitted'] += 1
                else:
                    rejected_duplicate += 1
            else:
                rejected_fitness += 1
        
        if self.generation % rebuild_interval == 0:
            self._rebuild_ngram()
        
        corpus_stats = self.corpus.stats()
        
        # Calculate rejection rates
        total_attempts = candidates_per_gen
        total_valid = len(candidates)
        validation_reject_rate = (total_attempts - total_valid) / total_attempts if total_attempts > 0 else 0
        
        return {
            'generation': self.generation,
            'attempts': total_attempts,
            'valid': total_valid,
            'validation_rejected': total_attempts - total_valid,
            'validation_reject_rate': validation_reject_rate,
            'fitness_rejected': rejected_fitness,
            'duplicate_rejected': rejected_duplicate,
            'admitted': admitted,
            'corpus_size': corpus_stats['size'],
            'avg_fitness': corpus_stats['avg_fitness'],
            'by_method': stats_by_method,
        }
    
    def get_best(self, n: int = 10) -> List[str]:
        """Get top n programs by fitness."""
        sorted_entries = sorted(self.corpus.entries, key=lambda e: e.fitness, reverse=True)
        return [e.program for e in sorted_entries[:n]]

# =============================================================================
# EXPANDED SEED CORPUS
# =============================================================================

SEED_CORPUS = [
    # Basic patterns with hyphens
    "(seq 60 - 64 - 67 -)",
    "(seq - 60 - 64 - 67)",
    "(seq 1 - 2 - 3 -)",
    "(seq - - 1 - - 2)",
    
    # Euclidean rhythms
    "(euclid 3 8)",
    "(euclid 5 8)",
    "(euclid 5 16)",
    "(euclid 7 16)",
    "(euclid (seq 60 64 67) 8)",
    "(euclid (seq 60 - 64 - 67) 16)",
    "(euclid (chord 60 0) 16)",
    "(euclid (seq 48 - 60 - 72) 8)",
    
    # Bjorklund
    "(bjork 3 8)",
    "(bjork 5 8)",
    "(bjork 7 16)",
    
    # Chords
    "(chord 60 0)",
    "(chord 60 1)",
    "(chord 65 0)",
    "(chord 67 4)",
    "(chord 48 5)",
    
    # Arpeggios
    "(arp (chord 60 0) 0 8)",
    "(arp (chord 60 1) 1 8)",
    "(arp (chord 60 4) 2 16)",
    "(arp (chord 67 0) 0 16)",
    
    # Ranges
    "(range 60 72)",
    "(range 48 60)",
    "(range 0 8)",
    "(range 60 72 2)",
    
    # Transforms
    "(reverse (seq 1 2 3 4))",
    "(reverse (seq 60 - 64 - 67))",
    "(palindrome (seq 1 2 3))",
    "(palindrome (seq 60 64 67))",
    "(mirror (seq 1 2 3))",
    "(rotate (seq 60 64 67 72) 1)",
    "(rotate (seq 1 - 2 - 3 -) 2)",
    
    # Interleave
    "(interleave (euclid 3 8) (euclid 5 8))",
    "(interleave (seq 60 64 67) (seq - - -))",
    "(interleave (bjork 3 8) (range 60 67))",
    
    # Selection
    "(take (range 0 12) 8)",
    "(take (seq 60 64 67 72 76) 3)",
    "(drop (range 60 72) 4)",
    "(every (range 60 84) 2)",
    "(slice (range 60 72) 2 6)",
    "(filter (seq 1 - 2 - 3))",
    
    # Probability/degradation
    "(prob (seq 60 64 67 72) 50)",
    "(prob (range 60 72) 75)",
    "(degrade (seq 1 2 3 4 5 6 7 8) 25)",
    "(degrade (euclid 5 8) 30)",
    "(maybe (seq 60 64 67) (seq - - -) 50)",
    
    # Drunk walk
    "(drunk 8 2 60)",
    "(drunk 16 3 48)",
    
    # Ramp
    "(ramp 60 72 8)",
    "(ramp 48 72 16)",
    
    # Cycle
    "(cycle (seq 60 - 64 -) 16)",
    "(cycle (seq 1 2 3) 8)",
    
    # Gate/delay
    "(gate (seq 60 64 67 72 76 79) 2)",
    "(delay (seq 60 64 67) 4)",
    "(delay (euclid 3 8) 2)",
    
    # Subdiv
    "(subdiv (seq 60 64 67) 2)",
    "(subdiv (euclid 3 8) 2)",
    
    # Quantize
    "(quantize (drunk 8 3 60) (seq 60 62 64 65 67 69 71 72))",
    "(quantize (range 58 70) (seq 60 62 64 65 67 69 71 72))",
    
    # Transpose
    "(transpose (seq 60 64 67) 12)",
    "(transpose (chord 60 0) -12)",
    
    # Grow
    "(grow (seq 1 2 3))",
    "(grow (seq 60 64 67))",
    
    # Shuffle
    "(shuffle (seq 60 64 67 72))",
    "(shuffle (range 60 72))",
    
    # Choose
    "(choose (seq 60 64 67) (seq 48 52 55))",
    "(choose (euclid 3 8) (bjork 5 8))",
    
    # Rep
    "(rep (seq 60 -) 4)",
    "(rep (seq 1 2 3) 2)",
    
    # Nested combinations
    "(seq (euclid 3 8) (euclid 5 8))",
    "(seq (chord 60 0) (chord 65 1))",
    "(reverse (euclid (seq 60 64 67) 8))",
    "(palindrome (euclid 3 8))",
    "(interleave (arp (chord 60 0) 0 8) (seq - - - -))",
    "(euclid (reverse (range 60 67)) 8)",
    "(rotate (euclid 5 8) 2)",
    "(prob (euclid (chord 60 0) 16) 75)",
    "(degrade (arp (chord 60 0) 0 16) 25)",
    "(seq (euclid 3 8) - - (euclid 5 8))",
    "(interleave (bjork 3 8) (bjork 5 8))",
    "(cycle (interleave (seq 60 -) (seq - 64)) 16)",
    "(quantize (drunk 16 2 64) (chord 60 0))",
    
    # Complex nested
    "(seq (arp (chord 60 0) 0 4) - - (arp (chord 65 1) 1 4))",
    "(interleave (euclid (seq 60 - 64) 8) (bjork 3 8))",
    "(reverse (interleave (seq 60 64 67) (seq - - -)))",
    "(palindrome (interleave (euclid 3 8) (seq 60 64 67)))",
    "(euclid (seq (chord 60 0) -) 16)",
    "(prob (interleave (euclid 3 8) (range 60 67)) 60)",
]

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run evolution and print interesting patterns."""
    print("=" * 60)
    print("ALIEN DSL PATTERN EVOLUTION")
    print("=" * 60)
    print()
    
    # Check for external validator
    if os.path.exists(ALIEN_PARSER_PATH):
        print(f"Using external validator: {ALIEN_PARSER_PATH}")
    else:
        print("Using internal validator (alien_parser not found)")
    
    # Load existing patterns from patterns.txt
    existing_patterns = load_patterns_file()
    if existing_patterns:
        print(f"Loaded {len(existing_patterns)} patterns from patterns.txt")
    
    # Combine seed corpus with existing patterns (deduped)
    combined_corpus = list(SEED_CORPUS)
    seen = set(SEED_CORPUS)
    for p in existing_patterns:
        if p not in seen:
            combined_corpus.append(p)
            seen.add(p)
    
    print(f"Starting with {len(combined_corpus)} total patterns ({len(SEED_CORPUS)} seeds + {len(existing_patterns)} from file)")
    print()
    
    evolver = AlienEvolver(combined_corpus, max_corpus_size=800)
    
    # Run evolution with rejection tracking
    num_generations = 100
    total_stats = {
        'attempts': 0,
        'valid': 0,
        'validation_rejected': 0,
        'fitness_rejected': 0,
        'duplicate_rejected': 0,
        'admitted': 0,
        'by_method': {
            'ngram': {'attempts': 0, 'valid': 0, 'admitted': 0},
            'mutate': {'attempts': 0, 'valid': 0, 'admitted': 0},
            'crossover': {'attempts': 0, 'valid': 0, 'admitted': 0},
            'random': {'attempts': 0, 'valid': 0, 'admitted': 0},
        }
    }
    
    for gen in range(num_generations):
        result = evolver.run_generation(
            candidates_per_gen=50,
            min_fitness=0.35,
            rebuild_interval=5
        )
        
        # Accumulate stats
        total_stats['attempts'] += result['attempts']
        total_stats['valid'] += result['valid']
        total_stats['validation_rejected'] += result['validation_rejected']
        total_stats['fitness_rejected'] += result['fitness_rejected']
        total_stats['duplicate_rejected'] += result['duplicate_rejected']
        total_stats['admitted'] += result['admitted']
        for method in ['ngram', 'mutate', 'crossover', 'random']:
            for key in ['attempts', 'valid', 'admitted']:
                total_stats['by_method'][method][key] += result['by_method'][method][key]
        
        if (gen + 1) % 20 == 0:
            print(f"Generation {result['generation']:3d}: "
                  f"corpus={result['corpus_size']:3d}, "
                  f"admitted={result['admitted']:2d}, "
                  f"valid={result['valid']}/{result['attempts']}, "
                  f"avg_fitness={result['avg_fitness']:.3f}")
    
    # Print rejection summary
    print()
    print("=" * 60)
    print("REJECTION STATISTICS")
    print("=" * 60)
    print(f"Total attempts:          {total_stats['attempts']}")
    print(f"Valid (passed parser):   {total_stats['valid']} ({100*total_stats['valid']/total_stats['attempts']:.1f}%)")
    print(f"Validation rejected:     {total_stats['validation_rejected']} ({100*total_stats['validation_rejected']/total_stats['attempts']:.1f}%)")
    print(f"Fitness rejected:        {total_stats['fitness_rejected']}")
    print(f"Duplicate rejected:      {total_stats['duplicate_rejected']}")
    print(f"Admitted to corpus:      {total_stats['admitted']}")
    print()
    print("By generation method:")
    for method in ['ngram', 'mutate', 'crossover', 'random']:
        m = total_stats['by_method'][method]
        valid_rate = 100 * m['valid'] / m['attempts'] if m['attempts'] > 0 else 0
        print(f"  {method:10s}: {m['attempts']:4d} attempts, {m['valid']:4d} valid ({valid_rate:.1f}%), {m['admitted']:4d} admitted")
    
    print()
    print("=" * 60)
    print("EVOLVED PATTERNS (validated)")
    print("=" * 60)
    print()
    
    # Get novel patterns not in original corpus
    novel_candidates = []
    for entry in sorted(evolver.corpus.entries, key=lambda e: e.fitness, reverse=True):
        if entry.program not in seen:
            # Double-check validation
            if evolver._validate(entry.program):
                novel_candidates.append(entry.program)
                if len(novel_candidates) >= 100:  # Get more candidates for diversity filtering
                    break
    
    # Filter for rest diversity first
    with_rests = ensure_rest_diversity(novel_candidates)
    print(f"Patterns with adequate rests: {len(with_rests)}/{len(novel_candidates)}")
    
    # Then filter for structural diversity
    diverse = filter_by_diversity(with_rests, max_patterns=30, similarity_threshold=0.5)
    print(f"Diverse patterns selected: {len(diverse)}")
    print()
    
    for program in diverse:
        print(program)
    
    # Append diverse patterns to patterns.txt
    if diverse:
        save_patterns_file(diverse, append=True)
        print()
        print(f"Appended {len(diverse)} diverse patterns to patterns.txt")
        print(f"Total patterns in file: {len(existing_patterns) + len(diverse)}")

if __name__ == '__main__':
    main()
