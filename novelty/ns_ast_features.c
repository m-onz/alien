/*
 * ns_ast_features - DSL expression → AST-shape BC vector for Pure Data
 *
 *   [ns_ast_features]
 *
 * Domain-agnostic projector that produces a behavioural-characterisation
 * vector from the *structure* of an alien DSL expression — not the
 * rendered output. Pairs naturally with ns_seq_features (which projects
 * the rendered sequence): two expressions that render to similar
 * sequences but have different operator structure get distinct AST
 * vectors, so the search rewards both phenotypic and genotypic novelty.
 *
 * Hot left inlet (anything / list / symbol):
 *     An alien DSL expression in either form:
 *       - atom-list:   e.g. clicked from a [msg] box
 *       - single symbol: e.g. as emitted by [ns_seq_propose] or [ns_corpus]
 *
 * Outlet:
 *     left: list of NS_AST_FEATURE_DIM floats — the AST-shape BC.
 *
 * Layout:
 *     0..(NS_OP_COUNT-1)  normalised operator-count histogram. Each slot
 *                         holds count(op_i) / total_op_nodes — i.e. the
 *                         fraction of operator nodes that use op_i. Sums
 *                         to 1.0 if any operators exist.
 *     NS_OP_COUNT + 0     size_norm     ast_size / NS_MAX_AST_SIZE
 *     NS_OP_COUNT + 1     depth_norm    ast_depth / NS_MAX_AST_DEPTH
 *     NS_OP_COUNT + 2     leaf_ratio    leaves / total_nodes
 *     NS_OP_COUNT + 3     avg_arity     mean child_count over op nodes
 *                                       (clamped to [0,1] via /max_arity)
 *
 * The whole vector is in [0, 1] by construction. No knowledge of MIDI,
 * pitch, rhythm, or value semantics — operates entirely on AST topology.
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"
#include "ns_alien_ast.h"
#include "../alien_core.h"

#include <string.h>
#include <stdio.h>

/* ======================================================================== */

#define NS_AST_STRUCT_DIMS 4
#define NS_AST_FEATURE_DIM ((int)NS_OP_COUNT + NS_AST_STRUCT_DIMS)
#define NS_AST_BUFSIZE 4096
/* Used for avg_arity normalisation. NS_OPS[]'s widest entry is 6
 * (scale, choose-up-to-4, drunk-with-bounds). Cap at 8 for headroom. */
#define NS_AST_MAX_ARITY 8.0f

static t_class *ns_ast_features_class;

typedef struct _ns_ast_features {
    t_object x_obj;
    t_outlet *x_out;
} t_ns_ast_features;

/* ======================================================================== */
/* AST WALKING                                                              */
/* ======================================================================== */

typedef struct {
    int op_counts[NS_OP_COUNT];   /* index aligns with NS_OPS[] */
    int total_op_nodes;
    int total_leaves;
    int total_nodes;
    long arity_sum;               /* sum of child_count over op nodes */
} ast_stats_t;

/* Walk every node in the tree, counting operator types and leaves.
 * Returns the index in NS_OPS[] for an operator NodeType, or -1 if
 * no match (defensive — every NodeType in the tree should appear in
 * the NS_OPS table). */
static int ns_op_index(NodeType t) {
    for (size_t i = 0; i < NS_OP_COUNT; i++) {
        if (NS_OPS[i].type == t) return (int)i;
    }
    return -1;
}

static void ast_walk(const ASTNode *n, ast_stats_t *s) {
    if (!n) return;
    s->total_nodes++;
    if (n->type == NODE_NUMBER || n->type == NODE_HYPHEN) {
        s->total_leaves++;
        return;
    }
    int idx = ns_op_index(n->type);
    if (idx >= 0) s->op_counts[idx]++;
    s->total_op_nodes++;
    s->arity_sum += n->data.op.child_count;
    for (int i = 0; i < n->data.op.child_count; i++) {
        ast_walk(n->data.op.children[i], s);
    }
}

/* ======================================================================== */
/* PROJECTION                                                               */
/* ======================================================================== */

static void emit_features(t_ns_ast_features *x, const ASTNode *root) {
    t_atom out[NS_AST_FEATURE_DIM];
    /* Default: zero vector — used when parse fails or tree is empty. */
    for (int i = 0; i < NS_AST_FEATURE_DIM; i++) {
        SETFLOAT(&out[i], 0.0f);
    }

    if (root) {
        ast_stats_t s = {0};
        for (size_t i = 0; i < NS_OP_COUNT; i++) s.op_counts[i] = 0;
        ast_walk(root, &s);

        /* Operator-count histogram — normalised to sum to 1. */
        if (s.total_op_nodes > 0) {
            float inv = 1.0f / (float)s.total_op_nodes;
            for (size_t i = 0; i < NS_OP_COUNT; i++) {
                SETFLOAT(&out[i], (t_float)s.op_counts[i] * inv);
            }
        }

        /* Structural metrics — all in [0, 1] by construction. */
        int size = ns_ast_size(root);
        int depth = ns_ast_depth(root);
        float size_norm = (float)size / (float)NS_MAX_AST_SIZE;
        if (size_norm > 1.0f) size_norm = 1.0f;
        float depth_norm = (float)depth / (float)NS_MAX_AST_DEPTH;
        if (depth_norm > 1.0f) depth_norm = 1.0f;
        float leaf_ratio = (s.total_nodes > 0)
            ? (float)s.total_leaves / (float)s.total_nodes : 0.0f;
        float avg_arity = (s.total_op_nodes > 0)
            ? ((float)s.arity_sum / (float)s.total_op_nodes) / NS_AST_MAX_ARITY
            : 0.0f;
        if (avg_arity > 1.0f) avg_arity = 1.0f;

        SETFLOAT(&out[NS_OP_COUNT + 0], (t_float)size_norm);
        SETFLOAT(&out[NS_OP_COUNT + 1], (t_float)depth_norm);
        SETFLOAT(&out[NS_OP_COUNT + 2], (t_float)leaf_ratio);
        SETFLOAT(&out[NS_OP_COUNT + 3], (t_float)avg_arity);
    }

    outlet_list(x->x_out, &s_list, NS_AST_FEATURE_DIM, out);
}

/* ======================================================================== */
/* INPUT — same atoms-to-DSL pattern as ns_seq_propose / ns_seq_info        */
/* ======================================================================== */

static void atoms_to_dsl(t_symbol *s, int argc, t_atom *argv,
                         char *buf, int max) {
    int pos = 0;
    if (s && s->s_name && s->s_name[0] != '\0' &&
        s != &s_list && s != &s_symbol && s != &s_float && s != &s_bang) {
        int w = snprintf(buf + pos, max - pos, "%s", s->s_name);
        if (w > 0 && w < max - pos) pos += w;
    }
    for (int i = 0; i < argc && pos < max - 1; i++) {
        if (pos > 0 && pos < max - 1) buf[pos++] = ' ';
        if (argv[i].a_type == A_FLOAT) {
            float f = atom_getfloat(&argv[i]);
            int w;
            if (f == (int)f) w = snprintf(buf + pos, max - pos, "%d", (int)f);
            else             w = snprintf(buf + pos, max - pos, "%g", f);
            if (w > 0 && w < max - pos) pos += w;
        } else if (argv[i].a_type == A_SYMBOL) {
            int w = snprintf(buf + pos, max - pos, "%s",
                             atom_getsymbol(&argv[i])->s_name);
            if (w > 0 && w < max - pos) pos += w;
        }
    }
    buf[pos < max ? pos : max - 1] = '\0';
}

static void ns_ast_features_anything(t_ns_ast_features *x, t_symbol *s,
                                      int argc, t_atom *argv) {
    char buf[NS_AST_BUFSIZE];
    atoms_to_dsl(s, argc, argv, buf, NS_AST_BUFSIZE);
    if (buf[0] == '\0') {
        emit_features(x, NULL);
        return;
    }
    Token *tokens = (Token *)getbytes(sizeof(Token) * 2048);
    if (!tokens) {
        emit_features(x, NULL);
        return;
    }
    int n_tok = tokenize(buf, tokens, 2048);
    if (n_tok < 0) {
        freebytes(tokens, sizeof(Token) * 2048);
        emit_features(x, NULL);
        return;
    }
    ASTNode *root = parse(tokens, n_tok);
    freebytes(tokens, sizeof(Token) * 2048);
    if (!root) {
        emit_features(x, NULL);
        return;
    }
    emit_features(x, root);
    ast_free(root);
}

static void ns_ast_features_list(t_ns_ast_features *x, t_symbol *s,
                                  int argc, t_atom *argv) {
    (void)s;
    ns_ast_features_anything(x, &s_list, argc, argv);
}

static void ns_ast_features_symbol(t_ns_ast_features *x, t_symbol *s) {
    if (!s) { emit_features(x, NULL); return; }
    ns_ast_features_anything(x, s, 0, NULL);
}

/* ======================================================================== */
/* CONSTRUCTOR / SETUP                                                      */
/* ======================================================================== */

static void *ns_ast_features_new(void) {
    t_ns_ast_features *x = (t_ns_ast_features *)pd_new(ns_ast_features_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}

void ns_ast_features_setup(void) {
    ns_ast_features_class = class_new(gensym("ns_ast_features"),
        (t_newmethod)ns_ast_features_new,
        0,
        sizeof(t_ns_ast_features),
        CLASS_DEFAULT,
        0);

    class_addanything(ns_ast_features_class, ns_ast_features_anything);
    class_addlist(ns_ast_features_class, ns_ast_features_list);
    class_addsymbol(ns_ast_features_class, ns_ast_features_symbol);

    post("ns_ast_features %s - alien DSL expr → %d-dim AST-shape BC",
         NS_VERSION_STRING, NS_AST_FEATURE_DIM);
}
