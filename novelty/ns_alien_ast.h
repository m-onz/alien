/*
 * ns_alien_ast.h - AST manipulation for alien DSL: mutation + crossover
 *
 * Header-only library that ports the AST-level evolutionary operators from
 * novelty_search.py (mutate, crossover, gen_tree, gen_op) onto the existing
 * ASTNode/Token/Parser machinery in alien_core.h.
 *
 * Public surface:
 *   ns_ast_render(node, buf, len)   serialize AST → string
 *   ns_ast_copy(node)               deep copy
 *   ns_ast_size(node)               count of nodes
 *   ns_ast_depth(node)              maximum nesting depth
 *   ns_ast_count_subtrees(node)     count of (sub)trees for crossover sampling
 *   ns_ast_get_subtree(node, idx)   walk to the idx'th subtree
 *
 *   ns_gen_tree(rng, depth_budget, require_seq)   random AST
 *   ns_mutate(node, rng, rate, depth)             recursive mutation
 *   ns_crossover(a, b, rng)                       subtree graft from b into a
 *
 * Random integers come from an ns_rng_t (defined in ns_core.h) so the
 * search is deterministic given a seed.
 *
 * Operator metadata is encoded in NS_OPS[]: name, NodeType, min_args,
 * max_args, and per-arg type constraints. NS_OP_GROUPS[] is the
 * "sibling-operator-swap" table used by mutate.
 */

#ifndef NS_ALIEN_AST_H
#define NS_ALIEN_AST_H

#include "ns_core.h"
#include "../alien_core.h"

#include <string.h>
#include <stdio.h>

/* ======================================================================== */
/* OPERATOR METADATA                                                        */
/* ======================================================================== */

typedef enum {
    NS_ARG_ANY = 0,    /* any expression — value or sub-pattern */
    NS_ARG_INT = 1,    /* must be a literal number */
    NS_ARG_SEQ = 2,    /* must be a sub-pattern (not a bare number/rest) */
} ns_arg_kind_t;

typedef struct {
    const char *name;
    NodeType type;
    int min_args;
    int max_args;
    int arg_kinds_len;
    ns_arg_kind_t arg_kinds[6];   /* index i → kind for child i; last repeats */
} ns_op_spec_t;

/* Mirrors novelty_search.py OPERATORS exactly. */
static const ns_op_spec_t NS_OPS[] = {
    {"seq",        NODE_SEQ,        0, 8, 1, {NS_ARG_ANY}},
    {"rep",        NODE_REP,        2, 2, 2, {NS_ARG_ANY, NS_ARG_INT}},
    {"add",        NODE_ADD,        2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"sub",        NODE_SUB,        2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"mul",        NODE_MUL,        2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"mod",        NODE_MOD,        2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"scale",      NODE_SCALE,      5, 5, 5, {NS_ARG_SEQ, NS_ARG_INT, NS_ARG_INT, NS_ARG_INT, NS_ARG_INT}},
    {"clamp",      NODE_CLAMP,      3, 3, 3, {NS_ARG_SEQ, NS_ARG_INT, NS_ARG_INT}},
    {"wrap",       NODE_WRAP,       3, 3, 3, {NS_ARG_SEQ, NS_ARG_INT, NS_ARG_INT}},
    {"fold",       NODE_FOLD,       3, 3, 3, {NS_ARG_SEQ, NS_ARG_INT, NS_ARG_INT}},
    {"euclid",     NODE_EUCLID,     2, 4, 4, {NS_ARG_ANY, NS_ARG_INT, NS_ARG_INT, NS_ARG_INT}},
    {"subdiv",     NODE_SUBDIV,     2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"reverse",    NODE_REVERSE,    1, 1, 1, {NS_ARG_SEQ}},
    {"rotate",     NODE_ROTATE,     2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"interleave", NODE_INTERLEAVE, 2, 2, 2, {NS_ARG_SEQ, NS_ARG_SEQ}},
    {"shuffle",    NODE_SHUFFLE,    1, 1, 1, {NS_ARG_SEQ}},
    {"mirror",     NODE_MIRROR,     1, 1, 1, {NS_ARG_SEQ}},
    {"take",       NODE_TAKE,       2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"drop",       NODE_DROP,       2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"slice",      NODE_SLICE,      3, 3, 3, {NS_ARG_SEQ, NS_ARG_INT, NS_ARG_INT}},
    {"every",      NODE_EVERY,      2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"filter",     NODE_FILTER,     1, 1, 1, {NS_ARG_SEQ}},
    {"range",      NODE_RANGE,      2, 3, 3, {NS_ARG_INT, NS_ARG_INT, NS_ARG_INT}},
    {"ramp",       NODE_RAMP,       3, 3, 3, {NS_ARG_INT, NS_ARG_INT, NS_ARG_INT}},
    {"choose",     NODE_CHOOSE,     2, 4, 1, {NS_ARG_ANY}},
    {"rand",       NODE_RAND,       1, 3, 3, {NS_ARG_INT, NS_ARG_INT, NS_ARG_INT}},
    {"prob",       NODE_PROB,       2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"drunk",      NODE_DRUNK,      3, 5, 5, {NS_ARG_INT, NS_ARG_INT, NS_ARG_INT, NS_ARG_INT, NS_ARG_INT}},
    {"quantize",   NODE_QUANTIZE,   2, 2, 2, {NS_ARG_SEQ, NS_ARG_SEQ}},
    {"arp",        NODE_ARP,        3, 3, 3, {NS_ARG_SEQ, NS_ARG_INT, NS_ARG_INT}},
    {"cycle",      NODE_CYCLE,      2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"grow",       NODE_GROW,       1, 1, 1, {NS_ARG_SEQ}},
    {"gate",       NODE_GATE,       2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"speed",      NODE_SPEED,      2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
    {"mask",       NODE_MASK,       2, 2, 2, {NS_ARG_SEQ, NS_ARG_SEQ}},
    {"delay",      NODE_DELAY,      2, 2, 2, {NS_ARG_SEQ, NS_ARG_INT}},
};
#define NS_OP_COUNT (sizeof(NS_OPS) / sizeof(NS_OPS[0]))

/* Look up a spec by NodeType. Linear scan — there are only ~36 entries. */
static inline const ns_op_spec_t *ns_op_lookup(NodeType t) {
    for (size_t i = 0; i < NS_OP_COUNT; i++) {
        if (NS_OPS[i].type == t) return &NS_OPS[i];
    }
    return NULL;
}

/* Resolve the expected arg_kind for a given child slot. Variadic ops (seq,
 * choose) have arg_kinds_len == 1 and the last entry repeats for all
 * positions; this helper encodes that convention. Returns NS_ARG_ANY if
 * spec is NULL or has no arg_kinds. */
static inline ns_arg_kind_t ns_op_arg_kind_at(const ns_op_spec_t *spec, int idx) {
    if (!spec || spec->arg_kinds_len <= 0 || idx < 0) return NS_ARG_ANY;
    int k = (idx < spec->arg_kinds_len) ? idx : spec->arg_kinds_len - 1;
    return spec->arg_kinds[k];
}

/* Operator-group sibling tables — used by mutate's "swap to similar op". */
typedef struct {
    NodeType members[8];
    int count;
} ns_op_group_t;

static const ns_op_group_t NS_OP_GROUPS[] = {
    {{NODE_ADD, NODE_SUB, NODE_MUL, NODE_MOD}, 4},                                 /* arith */
    {{NODE_CLAMP, NODE_WRAP, NODE_FOLD}, 3},                                       /* bound */
    {{NODE_EUCLID, NODE_SUBDIV, NODE_GATE, NODE_SPEED, NODE_MASK}, 5},             /* rhythm */
    {{NODE_REVERSE, NODE_ROTATE, NODE_SHUFFLE, NODE_MIRROR}, 4},                   /* list */
    {{NODE_TAKE, NODE_DROP, NODE_SLICE, NODE_EVERY, NODE_FILTER}, 5},              /* select */
    {{NODE_RANGE, NODE_RAMP, NODE_RAND, NODE_DRUNK}, 4},                           /* gen */
    {{NODE_CHOOSE, NODE_PROB, NODE_RAND, NODE_DRUNK}, 4},                          /* random */
    {{NODE_CYCLE, NODE_DELAY, NODE_GATE, NODE_SPEED}, 4},                          /* time */
    {{NODE_QUANTIZE, NODE_ARP, NODE_MIRROR}, 3},                                   /* musical */
    {{NODE_SEQ, NODE_REP, NODE_CYCLE, NODE_GROW}, 4},                              /* struct */
};
#define NS_OP_GROUP_COUNT (sizeof(NS_OP_GROUPS) / sizeof(NS_OP_GROUPS[0]))

/* Find a group containing this op; return its sibling pool. */
static inline const ns_op_group_t *ns_op_find_group(NodeType t) {
    for (size_t g = 0; g < NS_OP_GROUP_COUNT; g++) {
        for (int i = 0; i < NS_OP_GROUPS[g].count; i++) {
            if (NS_OP_GROUPS[g].members[i] == t) return &NS_OP_GROUPS[g];
        }
    }
    return NULL;
}

/* Bounds — these define how big and deep a generated AST can be before the
 * proposer rejects the offspring. Defaults are larger than novelty_search.py's
 * (40 / 4) because the Pd feedback-loop runs much longer lineages and needs
 * headroom to grow without hitting the cap immediately. The user can override
 * either via "max_size" / "max_depth" messages on ns_seq_propose. */
#define NS_MAX_AST_SIZE 80
#define NS_MAX_AST_DEPTH 12

static inline int ns_ast_is_leaf(const ASTNode *n) {
    return n && (n->type == NODE_NUMBER || n->type == NODE_HYPHEN);
}

/* ======================================================================== */
/* SERIALIZE — write an AST back to its source-form string.                 */
/* ======================================================================== */

static int ns_ast_render_internal(const ASTNode *n, char *buf, int *pos, int max) {
    if (!n || *pos >= max - 1) return 0;
    if (n->type == NODE_HYPHEN) {
        if (*pos + 1 >= max) return 0;
        buf[(*pos)++] = '-';
        return 1;
    }
    if (n->type == NODE_NUMBER) {
        /* alien's parser rejects negative literals. The producer should
         * never produce them, but we clamp at the render boundary too —
         * any future bug that lets a negative slip through will at worst
         * become a 0 in the rendered output, not an "Invalid character"
         * error from the evaluator. */
        int v = n->data.number;
        if (v < 0) v = 0;
        int written = snprintf(buf + *pos, max - *pos, "%d", v);
        if (written < 0 || written >= max - *pos) return 0;
        *pos += written;
        return 1;
    }
    const ns_op_spec_t *spec = ns_op_lookup(n->type);
    if (!spec) return 0;
    if (*pos + 1 >= max) return 0;
    buf[(*pos)++] = '(';
    int written = snprintf(buf + *pos, max - *pos, "%s", spec->name);
    if (written < 0 || written >= max - *pos) return 0;
    *pos += written;
    for (int i = 0; i < n->data.op.child_count; i++) {
        if (*pos + 1 >= max) return 0;
        buf[(*pos)++] = ' ';
        if (!ns_ast_render_internal(n->data.op.children[i], buf, pos, max)) return 0;
    }
    if (*pos + 1 >= max) return 0;
    buf[(*pos)++] = ')';
    return 1;
}

/* Returns 1 on success, 0 if the buffer would overflow. Always null-terminates. */
static inline int ns_ast_render(const ASTNode *n, char *buf, int max) {
    if (!buf || max <= 0) return 0;
    int pos = 0;
    int ok = ns_ast_render_internal(n, buf, &pos, max);
    buf[pos < max ? pos : max - 1] = '\0';
    return ok;
}

/* ======================================================================== */
/* COPY / SIZE / DEPTH                                                      */
/* ======================================================================== */

static inline ASTNode *ns_ast_copy(const ASTNode *n) {
    if (!n) return NULL;
    if (n->type == NODE_NUMBER) return ast_new_number(n->data.number);
    if (n->type == NODE_HYPHEN) return ast_new_hyphen();
    ASTNode *out = ast_new_op(n->type);
    if (!out) return NULL;
    for (int i = 0; i < n->data.op.child_count; i++) {
        ASTNode *child = ns_ast_copy(n->data.op.children[i]);
        if (!child || !ast_add_child(out, child)) {
            ast_free(out);
            if (child) ast_free(child);
            return NULL;
        }
    }
    return out;
}

static inline int ns_ast_size(const ASTNode *n) {
    if (!n) return 0;
    if (ns_ast_is_leaf(n)) return 1;
    int s = 1;
    for (int i = 0; i < n->data.op.child_count; i++) {
        s += ns_ast_size(n->data.op.children[i]);
    }
    return s;
}

static inline int ns_ast_depth(const ASTNode *n) {
    if (!n) return 0;
    if (ns_ast_is_leaf(n)) return 0;
    if (n->data.op.child_count == 0) return 1;
    int max_d = 0;
    for (int i = 0; i < n->data.op.child_count; i++) {
        int d = ns_ast_depth(n->data.op.children[i]);
        if (d > max_d) max_d = d;
    }
    return 1 + max_d;
}

/* ======================================================================== */
/* SUBTREE WALKING — for crossover, which picks a random subtree node       */
/* ======================================================================== */

static inline int ns_ast_count_subtrees(const ASTNode *n) {
    return ns_ast_size(n);  /* every node is itself a subtree */
}

/* Visit nodes in pre-order, returning a pointer to the idx'th, or NULL.
 * Also writes the parent pointer (or NULL if root) and the child slot index
 * within that parent into out_parent / out_slot when non-NULL. */
static inline int ns_ast_get_subtree_internal(ASTNode *n, ASTNode *parent, int parent_slot,
                                               int target, int *cur,
                                               ASTNode **out_parent, int *out_slot) {
    if (!n) return 0;
    if (*cur == target) {
        if (out_parent) *out_parent = parent;
        if (out_slot) *out_slot = parent_slot;
        return 1;  /* found — caller can read n via *cur counter walk */
    }
    (*cur)++;
    if (ns_ast_is_leaf(n)) return 0;
    for (int i = 0; i < n->data.op.child_count; i++) {
        if (ns_ast_get_subtree_internal(n->data.op.children[i], n, i, target, cur,
                                        out_parent, out_slot)) {
            return 1;
        }
    }
    return 0;
}

/* Returns the idx'th subtree pointer (pre-order). Returns NULL if out of range. */
static inline ASTNode *ns_ast_get_subtree(ASTNode *root, int idx,
                                          ASTNode **out_parent, int *out_slot) {
    if (!root || idx < 0) return NULL;
    int total = ns_ast_count_subtrees(root);
    if (idx >= total) return NULL;
    /* Walk to idx'th node by depth-first pre-order. */
    int cur = 0;
    ASTNode *parent = NULL;
    int slot = -1;
    /* We need to find both the node AND its parent simultaneously. Re-walk. */
    ASTNode *stack[64];
    ASTNode *parent_stack[64];
    int slot_stack[64];
    int sp = 0;
    stack[sp] = root;
    parent_stack[sp] = NULL;
    slot_stack[sp] = -1;
    sp = 1;
    int counter = 0;
    while (sp > 0) {
        sp--;
        ASTNode *cur_n = stack[sp];
        ASTNode *cur_p = parent_stack[sp];
        int cur_s = slot_stack[sp];
        if (counter == idx) {
            if (out_parent) *out_parent = cur_p;
            if (out_slot) *out_slot = cur_s;
            return cur_n;
        }
        counter++;
        if (!ns_ast_is_leaf(cur_n)) {
            /* Push children in reverse so leftmost is popped next (pre-order). */
            for (int i = cur_n->data.op.child_count - 1; i >= 0; i--) {
                if (sp >= 64) break;
                stack[sp] = cur_n->data.op.children[i];
                parent_stack[sp] = cur_n;
                slot_stack[sp] = i;
                sp++;
            }
        }
    }
    (void)cur; (void)parent; (void)slot;
    return NULL;
}

/* ======================================================================== */
/* RANDOM HELPERS                                                           */
/* ======================================================================== */

static inline int ns_rng_int_range(ns_rng_t *r, int min_inclusive, int max_inclusive) {
    if (max_inclusive < min_inclusive) return min_inclusive;
    int range = max_inclusive - min_inclusive + 1;
    return min_inclusive + (int)(ns_rng_uniform(r) * (float)range);
}

static inline int ns_rng_choice_int(ns_rng_t *r, const int *arr, int n) {
    if (n <= 0) return 0;
    return arr[ns_rng_int_range(r, 0, n - 1)];
}

/* Musical integer sampling — mirrors musical_int() in novelty_search.py. */
static inline int ns_musical_int(ns_rng_t *r, const char *kind) {
    static const int MIDI[] = {36, 38, 42, 48, 52, 55, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 79};
    static const int STEPS[] = {4, 6, 8, 12, 16};
    static const int HITS[] = {2, 3, 4, 5, 7};
    static const int SMALL[] = {1, 2, 3, 4, 5};
    static const int PROB_VALS[] = {20, 35, 50, 65, 80};
    /* Transpose-down is the `sub` operator's job — alien's parser rejects
     * negative literals, so this pool stays positive-only. The producer
     * picks `add` or `sub` based on direction, then chooses magnitude
     * from this pool. */
    static const int INTERVAL[] = {3, 5, 7, 12};

    if (kind && strcmp(kind, "midi") == 0)
        return ns_rng_choice_int(r, MIDI, sizeof(MIDI)/sizeof(int));
    if (kind && strcmp(kind, "steps") == 0)
        return ns_rng_choice_int(r, STEPS, sizeof(STEPS)/sizeof(int));
    if (kind && strcmp(kind, "hits") == 0)
        return ns_rng_choice_int(r, HITS, sizeof(HITS)/sizeof(int));
    if (kind && strcmp(kind, "small") == 0)
        return ns_rng_choice_int(r, SMALL, sizeof(SMALL)/sizeof(int));
    if (kind && strcmp(kind, "prob") == 0)
        return ns_rng_choice_int(r, PROB_VALS, sizeof(PROB_VALS)/sizeof(int));
    if (kind && strcmp(kind, "interval") == 0)
        return ns_rng_choice_int(r, INTERVAL, sizeof(INTERVAL)/sizeof(int));

    /* generic mix — same proportions as the Python version */
    float u = ns_rng_uniform(r);
    if (u < 0.35f) return ns_rng_choice_int(r, MIDI, sizeof(MIDI)/sizeof(int));
    if (u < 0.55f) return ns_rng_choice_int(r, STEPS, sizeof(STEPS)/sizeof(int));
    if (u < 0.75f) return ns_rng_choice_int(r, SMALL, sizeof(SMALL)/sizeof(int));
    if (u < 0.85f) return ns_rng_choice_int(r, INTERVAL, sizeof(INTERVAL)/sizeof(int));
    return ns_rng_int_range(r, 0, 100);
}

/* Forward declarations — circular references between tree generators. */
static ASTNode *ns_gen_tree(ns_rng_t *r, int depth_budget, int require_seq);
static ASTNode *ns_gen_op(ns_rng_t *r, NodeType op, int depth);

/* Leaf: a number or a rest. */
static ASTNode *ns_gen_leaf(ns_rng_t *r, float rest_prob) {
    if (ns_rng_uniform(r) < rest_prob) return ast_new_hyphen();
    return ast_new_number(ns_musical_int(r, "midi"));
}

/* Sequence literal of given length filled with leaves. */
static ASTNode *ns_gen_seq_literal(ns_rng_t *r, int length, float rest_prob) {
    ASTNode *seq = ast_new_op(NODE_SEQ);
    if (!seq) return NULL;
    for (int i = 0; i < length; i++) {
        ASTNode *leaf = ns_gen_leaf(r, rest_prob);
        if (!leaf || !ast_add_child(seq, leaf)) {
            ast_free(seq);
            if (leaf) ast_free(leaf);
            return NULL;
        }
    }
    return seq;
}

/* Pick one of several musical scales as a literal seq. */
static ASTNode *ns_gen_scale_literal(ns_rng_t *r) {
    static const int MAJOR[]      = {0, 2, 4, 5, 7, 9, 11};
    static const int MINOR[]      = {0, 2, 3, 5, 7, 8, 10};
    static const int BLUES[]      = {0, 3, 5, 6, 7, 10};
    static const int PENT[]       = {0, 2, 4, 7, 9};
    static const int LOCRIAN[]    = {0, 1, 3, 5, 6, 8, 10};
    static const int TRIAD[]      = {0, 4, 7};
    static const int *scales[]    = {MAJOR, MINOR, BLUES, PENT, LOCRIAN, TRIAD};
    static const int scale_lens[] = {7, 7, 6, 5, 7, 3};
    int idx = ns_rng_int_range(r, 0, 5);
    ASTNode *seq = ast_new_op(NODE_SEQ);
    if (!seq) return NULL;
    for (int i = 0; i < scale_lens[idx]; i++) {
        ASTNode *n = ast_new_number(scales[idx][i]);
        if (!n || !ast_add_child(seq, n)) {
            ast_free(seq);
            if (n) ast_free(n);
            return NULL;
        }
    }
    return seq;
}

/* Helpers: an "any" or "seq" argument with a depth budget. */
static ASTNode *ns_gen_any_arg(ns_rng_t *r, int rd) {
    if (rd <= 0 || ns_rng_uniform(r) < 0.45f) return ns_gen_leaf(r, 0.15f);
    return ns_gen_tree(r, rd - 1, 0);
}
static ASTNode *ns_gen_seq_arg(ns_rng_t *r, int rd) {
    if (rd <= 0 || ns_rng_uniform(r) < 0.35f)
        return ns_gen_seq_literal(r, ns_rng_int_range(r, 3, 6), 0.25f);
    return ns_gen_tree(r, rd - 1, 1);
}

/* ======================================================================== */
/* gen_op — produce a fresh, valid instance of a given operator             */
/* ======================================================================== */

static ASTNode *ns_gen_op(ns_rng_t *r, NodeType op, int depth) {
    int rd = NS_MAX_AST_DEPTH - depth;

    /* Helper to attach k children produced by callbacks. */
    #define OP(node_t) ASTNode *n = ast_new_op(node_t); if (!n) return NULL;
    #define ADD(child) do { ASTNode *_c = (child); if (!_c || !ast_add_child(n, _c)) { ast_free(n); if (_c) ast_free(_c); return NULL; } } while (0)
    #define INT(v) ast_new_number(v)

    if (op == NODE_SEQ) {
        int k = ns_rng_int_range(r, 2, 6);
        OP(NODE_SEQ);
        for (int i = 0; i < k; i++) {
            if (ns_rng_uniform(r) < 0.7f) ADD(ns_gen_leaf(r, 0.3f));
            else ADD(ns_gen_any_arg(r, rd - 1));
        }
        return n;
    }
    if (op == NODE_REP) {
        int reps[] = {2, 3, 4, 6, 8, 16};
        OP(NODE_REP);
        ADD(ns_gen_any_arg(r, rd - 1));
        ADD(INT(ns_rng_choice_int(r, reps, 6)));
        return n;
    }
    if (op == NODE_ADD || op == NODE_SUB || op == NODE_MUL || op == NODE_MOD) {
        int val;
        if (op == NODE_MOD) {
            int mods[] = {3, 4, 5, 7, 8, 12};
            val = ns_rng_choice_int(r, mods, 6);
        } else if (op == NODE_ADD || op == NODE_SUB) {
            val = ns_musical_int(r, "interval");
        } else {
            val = ns_musical_int(r, "small");
        }
        OP(op);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(val));
        return n;
    }
    if (op == NODE_SCALE) {
        int max_choices[] = {10, 12, 100, 127};
        int span_choices[] = {7, 12, 24};
        int fmin = 0;
        int fmax = ns_rng_choice_int(r, max_choices, 4);
        int tmin = ns_musical_int(r, "midi");
        int tmax = tmin + ns_rng_choice_int(r, span_choices, 3);
        OP(NODE_SCALE);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(fmin)); ADD(INT(fmax)); ADD(INT(tmin)); ADD(INT(tmax));
        return n;
    }
    if (op == NODE_CLAMP || op == NODE_WRAP || op == NODE_FOLD) {
        int span_choices[] = {7, 12, 24};
        int offset_choices[] = {0, 4, 7};
        int lo = ns_musical_int(r, "midi") - ns_rng_choice_int(r, offset_choices, 3);
        int hi = lo + ns_rng_choice_int(r, span_choices, 3);
        OP(op);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(lo)); ADD(INT(hi));
        return n;
    }
    if (op == NODE_EUCLID) {
        OP(NODE_EUCLID);
        if (ns_rng_uniform(r) < 0.5f) ADD(INT(ns_musical_int(r, "hits")));
        else ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(ns_musical_int(r, "steps")));
        if (ns_rng_uniform(r) < 0.3f) {
            ADD(INT(ns_rng_int_range(r, 0, 7)));
            if (ns_rng_uniform(r) < 0.4f) ADD(INT(ns_musical_int(r, "midi")));
        }
        return n;
    }
    if (op == NODE_SUBDIV) {
        int divs[] = {2, 3, 4};
        OP(NODE_SUBDIV);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(ns_rng_choice_int(r, divs, 3)));
        return n;
    }
    if (op == NODE_REVERSE || op == NODE_SHUFFLE || op == NODE_MIRROR ||
        op == NODE_FILTER  || op == NODE_GROW) {
        OP(op);
        ADD(ns_gen_seq_arg(r, rd - 1));
        return n;
    }
    if (op == NODE_ROTATE) {
        OP(NODE_ROTATE);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(ns_rng_int_range(r, 1, 7)));
        return n;
    }
    if (op == NODE_INTERLEAVE) {
        OP(NODE_INTERLEAVE);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(ns_gen_seq_arg(r, rd - 1));
        return n;
    }
    if (op == NODE_TAKE || op == NODE_DROP || op == NODE_EVERY ||
        op == NODE_GATE || op == NODE_SPEED || op == NODE_DELAY || op == NODE_CYCLE) {
        int choices[] = {2, 3, 4, 6, 8};
        int small_choices[] = {2, 3, 4};
        int v = (op == NODE_EVERY || op == NODE_GATE)
            ? ns_rng_choice_int(r, small_choices, 3)
            : ns_rng_choice_int(r, choices, 5);
        OP(op);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(v));
        return n;
    }
    if (op == NODE_SLICE) {
        int start = ns_rng_int_range(r, 0, 3);
        int end = start + ns_rng_int_range(r, 2, 6);
        OP(NODE_SLICE);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(start)); ADD(INT(end));
        return n;
    }
    if (op == NODE_RANGE) {
        int starts = ns_musical_int(r, "midi");
        int span_choices[] = {5, 7, 12, 24};
        int end = starts + ns_rng_choice_int(r, span_choices, 4);
        OP(NODE_RANGE);
        ADD(INT(starts)); ADD(INT(end));
        if (ns_rng_uniform(r) < 0.3f) {
            int steps[] = {1, 2, 3};
            ADD(INT(ns_rng_choice_int(r, steps, 3)));
        }
        return n;
    }
    if (op == NODE_RAMP) {
        int starts = ns_musical_int(r, "midi");
        int dirs[] = {-12, -7, 7, 12, 24};
        int end = starts + ns_rng_choice_int(r, dirs, 5);
        OP(NODE_RAMP);
        ADD(INT(starts)); ADD(INT(end)); ADD(INT(ns_musical_int(r, "steps")));
        return n;
    }
    if (op == NODE_DRUNK) {
        int max_choices[] = {1, 2, 3, 5};
        int span_choices[] = {5, 7, 12};
        int steps = ns_musical_int(r, "steps");
        int mx = ns_rng_choice_int(r, max_choices, 4);
        int starts = ns_musical_int(r, "midi");
        OP(NODE_DRUNK);
        ADD(INT(steps)); ADD(INT(mx)); ADD(INT(starts));
        if (ns_rng_uniform(r) < 0.5f) {
            int lo = starts - ns_rng_choice_int(r, span_choices, 3);
            int hi = starts + ns_rng_choice_int(r, span_choices, 3);
            ADD(INT(lo)); ADD(INT(hi));
        }
        return n;
    }
    if (op == NODE_RAND) {
        int count = ns_musical_int(r, "steps");
        OP(NODE_RAND);
        ADD(INT(count));
        if (ns_rng_uniform(r) < 0.7f) {
            int span_choices[] = {7, 12, 24};
            int lo = ns_musical_int(r, "midi");
            int hi = lo + ns_rng_choice_int(r, span_choices, 3);
            ADD(INT(lo)); ADD(INT(hi));
        }
        return n;
    }
    if (op == NODE_PROB) {
        OP(NODE_PROB);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(ns_musical_int(r, "prob")));
        return n;
    }
    if (op == NODE_CHOOSE) {
        int k = ns_rng_int_range(r, 2, 3);
        OP(NODE_CHOOSE);
        for (int i = 0; i < k; i++) ADD(ns_gen_any_arg(r, rd - 1));
        return n;
    }
    if (op == NODE_QUANTIZE) {
        OP(NODE_QUANTIZE);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(ns_gen_scale_literal(r));
        return n;
    }
    if (op == NODE_ARP) {
        OP(NODE_ARP);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(INT(ns_rng_int_range(r, 0, 2)));
        ADD(INT(ns_musical_int(r, "steps")));
        return n;
    }
    if (op == NODE_MASK) {
        OP(NODE_MASK);
        ADD(ns_gen_seq_arg(r, rd - 1));
        ADD(ns_gen_seq_arg(r, rd - 1));
        return n;
    }

    /* Fallback. */
    return ns_gen_seq_literal(r, 4, 0.3f);

    #undef OP
    #undef ADD
    #undef INT
}

/* ======================================================================== */
/* gen_tree — top-level random tree, weighted toward musical roots          */
/* ======================================================================== */

static ASTNode *ns_gen_tree(ns_rng_t *r, int depth_budget, int require_seq) {
    if (depth_budget <= 0) {
        if (require_seq) return ns_gen_seq_literal(r, 4, 0.3f);
        return ns_gen_leaf(r, 0.15f);
    }
    /* Every alien operator appears at least once. The original Python bias
     * (4× euclid, 3× seq) skewed lineages too narrowly; here it's flattened
     * to a maximum of 2× so the full DSL palette is reachable from any
     * fresh-tree mutation. */
    static const NodeType OPS[] = {
        /* Common musical roots — slight bias (2×). */
        NODE_EUCLID, NODE_EUCLID,
        NODE_SEQ, NODE_SEQ,
        NODE_INTERLEAVE, NODE_INTERLEAVE,
        NODE_ARP, NODE_ARP,
        NODE_QUANTIZE, NODE_QUANTIZE,
        NODE_MASK, NODE_MASK,
        NODE_DRUNK, NODE_DRUNK,
        NODE_REP, NODE_REP,
        NODE_RANGE, NODE_RANGE,
        NODE_RAMP, NODE_RAMP,
        /* Everything else — uniform 1×. */
        NODE_ROTATE, NODE_MIRROR, NODE_PROB, NODE_SUBDIV, NODE_GATE,
        NODE_SPEED, NODE_CYCLE, NODE_REVERSE, NODE_GROW, NODE_CHOOSE,
        NODE_ADD, NODE_SUB, NODE_MUL, NODE_MOD,
        NODE_SCALE, NODE_CLAMP, NODE_WRAP, NODE_FOLD,
        NODE_TAKE, NODE_DROP, NODE_SLICE, NODE_EVERY, NODE_FILTER,
        NODE_RAND, NODE_DELAY, NODE_SHUFFLE,
    };
    int nops = sizeof(OPS) / sizeof(OPS[0]);
    NodeType chosen = OPS[ns_rng_int_range(r, 0, nops - 1)];
    (void)require_seq;  /* upper-level caller ignores if root */
    return ns_gen_op(r, chosen, NS_MAX_AST_DEPTH - depth_budget);
}

/* ======================================================================== */
/* MUTATE                                                                   */
/*                                                                          */
/* Returns a NEW AST. Original is unchanged. Caller must ast_free(new).     */
/* Decisions match novelty_search.py's mutate() exactly.                    */
/* ======================================================================== */

/* Helper: copy n.type but recursively mutate each child. */
static ASTNode *ns_mutate(ns_rng_t *r, const ASTNode *n, float rate, int depth);

static ASTNode *ns_mutate_recurse_children(ns_rng_t *r, const ASTNode *n, float rate, int depth) {
    ASTNode *out = ast_new_op(n->type);
    if (!out) return NULL;
    for (int i = 0; i < n->data.op.child_count; i++) {
        ASTNode *new_child = ns_mutate(r, n->data.op.children[i], rate, depth + 1);
        if (!new_child || !ast_add_child(out, new_child)) {
            ast_free(out);
            if (new_child) ast_free(new_child);
            return NULL;
        }
    }
    return out;
}

/* ------------------------------------------------------------------------ */
/* Leaf-escape: when a leaf is at the root, occasionally wrap it back into  */
/* a seq with a few extra leaves. Without this a lineage that has collapsed */
/* to a single number can never grow back. Returns NULL if alloc fails —    */
/* caller must fall through to standard numeric mutation.                   */
/* ------------------------------------------------------------------------ */

static ASTNode *ns_mutate_leaf_escape(ns_rng_t *r, const ASTNode *self_proto) {
    ASTNode *seq = ast_new_op(NODE_SEQ);
    if (!seq) return NULL;
    ASTNode *self = ns_ast_copy(self_proto);
    if (!self || !ast_add_child(seq, self)) {
        if (self) ast_free(self);
        ast_free(seq);
        return NULL;
    }
    int extra = ns_rng_int_range(r, 2, 4);
    for (int i = 0; i < extra; i++) {
        ASTNode *leaf = ns_gen_leaf(r, 0.25f);
        if (!leaf || !ast_add_child(seq, leaf)) {
            if (leaf) ast_free(leaf);
            break;
        }
    }
    return seq;
}

/* ------------------------------------------------------------------------ */
/* Leaf mutation: NUMBER stays NUMBER (delta or musical_int — never HYPHEN, */
/* which would break operators expecting int args). HYPHEN flips between    */
/* note and rest. At root, both have a 10% leaf-escape chance.              */
/* ------------------------------------------------------------------------ */

static ASTNode *ns_mutate_leaf(ns_rng_t *r, const ASTNode *n, int depth) {
    float c = ns_rng_uniform(r);

    if (depth == 0 && c < 0.10f) {
        ASTNode *escaped = ns_mutate_leaf_escape(r, n);
        if (escaped) return escaped;
        /* fall through if escape allocation failed */
    }

    if (n->type == NODE_NUMBER) {
        int new_val;
        if (c < 0.55f) {
            static const int DELTAS[] = {-12, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 12};
            new_val = n->data.number + ns_rng_choice_int(r, DELTAS, 12);
        } else {
            new_val = ns_musical_int(r, NULL);
        }
        /* alien's parser rejects negative literals — clamp to non-negative.
         * This is the last line of defence; ns_musical_int's pools are all
         * non-negative now too. */
        if (new_val < 0) new_val = 0;
        return ast_new_number(new_val);
    }

    /* NODE_HYPHEN: 0.5 threshold (not 0.55 like NUMBER). */
    if (c < 0.5f) return ast_new_number(ns_musical_int(r, "midi"));
    return ast_new_hyphen();
}

/* ------------------------------------------------------------------------ */
/* Sibling-op swap: change the operator type while preserving children.    */
/* Tries up to 8 candidate siblings. Requires both arity AND arg_kind      */
/* compatibility — without the kind check, swapping (subdiv seq int) with  */
/* (mask seq seq) lands an int in mask's seq slot, producing trees the     */
/* evaluator rejects. Falls back to recurse-children if no compatible      */
/* sibling found.                                                           */
/* ------------------------------------------------------------------------ */

/* Returns 1 if `child` would satisfy `kind` at its slot, 0 otherwise.
 * NS_ARG_ANY accepts anything; INT wants NODE_NUMBER; SEQ wants a non-leaf
 * (operator) — a bare number or rest can't be a seq argument. */
static inline int ns_child_satisfies_kind(const ASTNode *child, ns_arg_kind_t kind) {
    if (!child) return 0;
    switch (kind) {
        case NS_ARG_ANY: return 1;
        case NS_ARG_INT: return child->type == NODE_NUMBER;
        case NS_ARG_SEQ: return !ns_ast_is_leaf(child);
    }
    return 1;
}

static ASTNode *ns_mutate_swap_sibling(ns_rng_t *r, const ASTNode *n, float rate, int depth) {
    const ns_op_group_t *group = ns_op_find_group(n->type);
    if (group && group->count >= 2) {
        for (int tries = 0; tries < 8; tries++) {
            NodeType sib = group->members[ns_rng_int_range(r, 0, group->count - 1)];
            if (sib == n->type) continue;
            const ns_op_spec_t *spec_sib = ns_op_lookup(sib);
            if (!spec_sib) continue;
            if (n->data.op.child_count < spec_sib->min_args ||
                n->data.op.child_count > spec_sib->max_args) continue;

            /* Verify every existing child is kind-compatible with the
             * sibling's slot at that index. If even one mismatches, the
             * tree would fail evaluation — try a different sibling. */
            int kinds_ok = 1;
            for (int i = 0; i < n->data.op.child_count; i++) {
                ns_arg_kind_t want = ns_op_arg_kind_at(spec_sib, i);
                if (!ns_child_satisfies_kind(n->data.op.children[i], want)) {
                    kinds_ok = 0;
                    break;
                }
            }
            if (!kinds_ok) continue;

            ASTNode *out = ast_new_op(sib);
            if (!out) return NULL;
            int ok = 1;
            for (int i = 0; i < n->data.op.child_count; i++) {
                ASTNode *cc = ns_ast_copy(n->data.op.children[i]);
                if (!cc || !ast_add_child(out, cc)) {
                    if (cc) ast_free(cc);
                    ok = 0;
                    break;
                }
            }
            if (ok) return out;
            ast_free(out);
        }
    }
    return ns_mutate_recurse_children(r, n, rate, depth);
}

/* ------------------------------------------------------------------------ */
/* Add or remove a child. Bias 80/20 toward add. Added child respects the   */
/* parent's expected arg type so we never synthesise broken expressions.    */
/* ------------------------------------------------------------------------ */

static ASTNode *ns_mutate_add_remove(ns_rng_t *r, const ASTNode *n, float rate, int depth) {
    const ns_op_spec_t *spec = ns_op_lookup(n->type);
    ASTNode *out = ns_mutate_recurse_children(r, n, rate, depth);
    if (!out) return NULL;
    if (!spec) return out;

    int can_add = (out->data.op.child_count < spec->max_args);
    int can_remove = (out->data.op.child_count > spec->min_args);
    int do_add = (ns_rng_uniform(r) < 0.8f) && can_add;
    if (!do_add && !can_remove && can_add) do_add = 1;

    if (do_add) {
        int slot = out->data.op.child_count;
        int last = (spec->arg_kinds_len > 0) ? spec->arg_kinds_len - 1 : 0;
        int kind_idx = (slot < spec->arg_kinds_len) ? slot : last;
        ns_arg_kind_t kind = (spec->arg_kinds_len > 0)
            ? spec->arg_kinds[kind_idx] : NS_ARG_ANY;
        ASTNode *leaf = NULL;
        if (kind == NS_ARG_INT) {
            leaf = ast_new_number(ns_musical_int(r, NULL));
        } else if (kind == NS_ARG_SEQ) {
            leaf = ns_gen_seq_literal(r, ns_rng_int_range(r, 3, 5), 0.25f);
        } else {
            leaf = ns_gen_leaf(r, 0.3f);
        }
        if (leaf && !ast_add_child(out, leaf)) ast_free(leaf);
    } else if (can_remove) {
        int rm = ns_rng_int_range(r, 0, out->data.op.child_count - 1);
        ast_free(out->data.op.children[rm]);
        for (int i = rm; i < out->data.op.child_count - 1; i++) {
            out->data.op.children[i] = out->data.op.children[i + 1];
        }
        out->data.op.child_count--;
    }
    return out;
}

/* ------------------------------------------------------------------------ */
/* Wrap in another operator — the main growth path. If wrapping would push  */
/* past NS_MAX_AST_DEPTH, prune at root (30% chance, non-leaf children only)*/
/* or fall back to recurse. The 16-wrapper pool covers every alien op that  */
/* takes a seq as its first arg.                                            */
/* ------------------------------------------------------------------------ */

static ASTNode *ns_mutate_wrap(ns_rng_t *r, const ASTNode *n, float rate, int depth) {
    /* Depth-cap fallback: prune-at-root or recurse. */
    if (depth + ns_ast_depth(n) >= NS_MAX_AST_DEPTH) {
        if (depth == 0 && n->data.op.child_count > 0 &&
            ns_rng_uniform(r) < 0.30f) {
            for (int tries = 0; tries < 8; tries++) {
                int pick = ns_rng_int_range(r, 0, n->data.op.child_count - 1);
                ASTNode *cand = n->data.op.children[pick];
                if (!ns_ast_is_leaf(cand)) return ns_ast_copy(cand);
            }
        }
        return ns_mutate_recurse_children(r, n, rate, depth);
    }

    static const NodeType WRAPPERS[] = {
        NODE_REVERSE, NODE_MIRROR, NODE_SHUFFLE, NODE_GROW, NODE_FILTER,
        NODE_CYCLE, NODE_PROB, NODE_REP,
        NODE_ROTATE, NODE_SUBDIV, NODE_GATE, NODE_SPEED, NODE_DELAY,
        NODE_EVERY, NODE_TAKE, NODE_DROP,
    };
    int n_wrappers = sizeof(WRAPPERS) / sizeof(WRAPPERS[0]);
    NodeType w = WRAPPERS[ns_rng_int_range(r, 0, n_wrappers - 1)];

    ASTNode *inner = ns_ast_copy(n);
    if (!inner) return NULL;
    ASTNode *out = ast_new_op(w);
    if (!out) { ast_free(inner); return NULL; }
    if (!ast_add_child(out, inner)) { ast_free(out); ast_free(inner); return NULL; }

    /* Second arg if the wrapper needs one; sampled from sensible ranges. */
    ASTNode *second = NULL;
    if (w == NODE_CYCLE || w == NODE_DELAY) {
        second = ast_new_number(ns_musical_int(r, "steps"));
    } else if (w == NODE_PROB) {
        second = ast_new_number(ns_musical_int(r, "prob"));
    } else if (w == NODE_REP) {
        static const int REPS[] = {2, 3, 4, 6, 8};
        second = ast_new_number(ns_rng_choice_int(r, REPS, 5));
    } else if (w == NODE_ROTATE) {
        second = ast_new_number(ns_rng_int_range(r, 1, 7));
    } else if (w == NODE_SUBDIV) {
        static const int DIVS[] = {2, 3, 4};
        second = ast_new_number(ns_rng_choice_int(r, DIVS, 3));
    } else if (w == NODE_GATE || w == NODE_EVERY) {
        static const int SMALL[] = {2, 3, 4};
        second = ast_new_number(ns_rng_choice_int(r, SMALL, 3));
    } else if (w == NODE_SPEED) {
        static const int SP[] = {2, 3, 4, 6, 8};
        second = ast_new_number(ns_rng_choice_int(r, SP, 5));
    } else if (w == NODE_TAKE || w == NODE_DROP) {
        static const int N[] = {2, 3, 4, 5, 6};
        second = ast_new_number(ns_rng_choice_int(r, N, 5));
    }
    /* REVERSE, MIRROR, SHUFFLE, GROW, FILTER take 1 arg only. */
    if (second && !ast_add_child(out, second)) {
        ast_free(out); ast_free(second); return NULL;
    }
    return out;
}

/* ------------------------------------------------------------------------ */
/* Reorder children — swap two random child positions. Preserves shape and  */
/* depth; only changes child ordering for order-sensitive operators.        */
/*                                                                          */
/* Critical invariant: only swap positions of matching arg_kind. Without    */
/* this guard, mutating (rotate seq int) by swapping slots produces         */
/* (rotate int seq) which the evaluator rejects with "rotate: n must be     */
/* single number" — wasting the iteration. Variadic ops like seq/choose     */
/* (arg_kinds_len == 1) freely allow any swap because all positions share   */
/* the same kind. Fully-typed ops with mixed kinds (rotate, every, take,    */
/* drop, subdiv, ...) only swap when the random pair happens to land on     */
/* same-kind slots; otherwise we fall through to recurse-children, which    */
/* still does useful work via the recursive pass.                           */
/* ------------------------------------------------------------------------ */

static ASTNode *ns_mutate_reorder(ns_rng_t *r, const ASTNode *n, float rate, int depth) {
    if (n->data.op.child_count < 2) {
        return ns_mutate_recurse_children(r, n, rate, depth);
    }
    ASTNode *out = ns_mutate_recurse_children(r, n, rate, depth);
    if (!out) return NULL;

    const ns_op_spec_t *spec = ns_op_lookup(n->type);

    /* Try up to 8 random index pairs to find a kind-matched swap. If none
     * found, return out unchanged — children were still recursively mutated
     * by ns_mutate_recurse_children above, so the iteration isn't wasted. */
    for (int tries = 0; tries < 8; tries++) {
        int i = ns_rng_int_range(r, 0, out->data.op.child_count - 1);
        int j = ns_rng_int_range(r, 0, out->data.op.child_count - 1);
        if (i == j) continue;
        ns_arg_kind_t ki = ns_op_arg_kind_at(spec, i);
        ns_arg_kind_t kj = ns_op_arg_kind_at(spec, j);
        if (ki != kj) continue;
        ASTNode *tmp = out->data.op.children[i];
        out->data.op.children[i] = out->data.op.children[j];
        out->data.op.children[j] = tmp;
        return out;
    }
    return out;
}

/* ------------------------------------------------------------------------ */
/* Dispatcher: depth-decay rate gate, then for operator nodes a uniform-    */
/* random pick across six branches. Each branch is its own named function   */
/* with its own invariants; this dispatcher just routes.                    */
/*                                                                          */
/* Branch probabilities (operator nodes only):                              */
/*   0.00 .. 0.30   recurse-children   (30%)                                */
/*   0.30 .. 0.45   sibling-op swap    (15%, gated on depth+1 < cap)        */
/*   0.45 .. 0.55   add/remove child   (10%)                                */
/*   0.55 .. 0.95   wrap (or prune)    (40%, the main growth path)          */
/*   0.95 .. 1.00   reorder children   (5%)                                 */
/*                                                                          */
/* When sibling-swap is gated out by depth, we fall through to add/remove.  */
/* ------------------------------------------------------------------------ */

static ASTNode *ns_mutate(ns_rng_t *r, const ASTNode *n, float rate, int depth) {
    if (!n) return NULL;

    /* Depth-decay: at depth d, mutation rate is rate / (1 + 0.15·d).
     * Skip → copy leaf or recurse into operator children. */
    if (ns_rng_uniform(r) > rate / (1.0f + 0.15f * (float)depth)) {
        if (ns_ast_is_leaf(n)) return ns_ast_copy(n);
        return ns_mutate_recurse_children(r, n, rate, depth);
    }

    if (ns_ast_is_leaf(n)) return ns_mutate_leaf(r, n, depth);

    float c = ns_rng_uniform(r);
    if (c < 0.30f) return ns_mutate_recurse_children(r, n, rate, depth);
    if (c < 0.45f && depth + 1 < NS_MAX_AST_DEPTH)
        return ns_mutate_swap_sibling(r, n, rate, depth);
    if (c < 0.55f) return ns_mutate_add_remove(r, n, rate, depth);
    if (c < 0.95f) return ns_mutate_wrap(r, n, rate, depth);
    return ns_mutate_reorder(r, n, rate, depth);
}

/* ======================================================================== */
/* CROSSOVER — graft a random subtree from b into a random cut of a.        */
/* Returns NEW tree. Originals unchanged. Caller frees the result.          */
/* ======================================================================== */

static inline ASTNode *ns_crossover(const ASTNode *a, const ASTNode *b, ns_rng_t *r) {
    if (!a) return ns_ast_copy(b);
    if (!b) return ns_ast_copy(a);

    ASTNode *a_copy = ns_ast_copy(a);
    if (!a_copy) return NULL;

    int a_count = ns_ast_count_subtrees(a_copy);
    int b_count = ns_ast_count_subtrees(b);
    if (a_count <= 0 || b_count <= 0) return a_copy;

    int tgt_idx = ns_rng_int_range(r, 0, a_count - 1);
    int donor_idx = ns_rng_int_range(r, 0, b_count - 1);

    ASTNode *target_parent = NULL;
    int target_slot = -1;
    ASTNode *target = ns_ast_get_subtree(a_copy, tgt_idx, &target_parent, &target_slot);
    if (!target) return a_copy;

    ASTNode *donor = ns_ast_get_subtree((ASTNode *)b, donor_idx, NULL, NULL);
    if (!donor) return a_copy;

    ASTNode *donor_copy = ns_ast_copy(donor);
    if (!donor_copy) return a_copy;

    if (target_parent == NULL) {
        ast_free(a_copy);
        return donor_copy;
    }
    ASTNode *old = target_parent->data.op.children[target_slot];
    target_parent->data.op.children[target_slot] = donor_copy;
    ast_free(old);
    return a_copy;
}

#endif /* NS_ALIEN_AST_H */
