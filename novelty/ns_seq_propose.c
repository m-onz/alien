/*
 * ns_seq_propose - AST-level evolutionary producer for alien DSL expressions
 *
 *   [ns_seq_propose]                    default: mutate, rate=0.2
 *   [ns_seq_propose mutate 0.2]
 *   [ns_seq_propose crossover]
 *   [ns_seq_propose random]
 *
 * Each list received on the hot inlet is parsed as an alien DSL expression
 * and pushed onto a 2-deep parent history (newest first). What happens next
 * depends on mode:
 *
 *   mutate:    apply ns_mutate to parent[0], output the result
 *   crossover: graft a random subtree from parent[1] into parent[0]
 *   random:    emit a freshly-generated tree (parent history ignored)
 *
 * Bang on the hot inlet re-applies the current mode to the cached parent(s)
 * with fresh randomness.
 *
 * The output is the proposed expression as a single symbol — directly
 * connectable to [alien]'s input.
 *
 * Hot left inlet (anything / list / symbol / bang):
 *     A parent expression in alien-compatible form. Both forms work:
 *       - atom-list:  the selector plus args, e.g. (seq 60 - 64 - 67 -)
 *         clicked from a [msg] box
 *       - single symbol: the entire expression text, e.g. as emitted by
 *         [alien_wrap]
 *     Bang re-proposes from the cached parents.
 *
 * Cold right inlet (proxy):
 *     rate <f>        set mutation rate (0..1)
 *     mode <m>        mutate | crossover | random
 *     seed <int>      reseed RNG (deterministic from here)
 *     max_size <n>    reject offspring with > n nodes; retry up to 10×
 *     max_depth <n>   reject offspring deeper than n; retry up to 10×
 *     clear           wipe parent history
 *
 * Outlet:
 *     left (anything): the proposed expression as a single symbol
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"
#include "ns_alien_ast.h"

#include <string.h>
#include <stdio.h>
#include <time.h>

/* ======================================================================== */

static t_class *ns_seq_propose_class;
static t_class *ns_seq_propose_proxy_class;

typedef struct _ns_seq_propose t_ns_seq_propose;

typedef struct _ns_seq_propose_proxy {
    t_pd p_pd;
    t_ns_seq_propose *p_owner;
} t_ns_seq_propose_proxy;

typedef enum {
    NS_PROPOSE_MUTATE = 0,
    NS_PROPOSE_CROSSOVER = 1,
    NS_PROPOSE_RANDOM = 2,
} ns_propose_mode_t;

#define NS_PROPOSE_BUFSIZE 4096

struct _ns_seq_propose {
    t_object x_obj;
    t_outlet *x_out;
    ns_propose_mode_t x_mode;
    float x_rate;
    int x_max_size;
    int x_max_depth;
    ns_rng_t x_rng;
    /* 2-deep parent history. NULL slot = empty. */
    char x_parent0[NS_PROPOSE_BUFSIZE];
    char x_parent1[NS_PROPOSE_BUFSIZE];
    int x_has_parent0;
    int x_has_parent1;
    t_ns_seq_propose_proxy x_proxy;
};

/* ======================================================================== */
/* HELPERS                                                                  */
/* ======================================================================== */

/* Convert an incoming Pd message (selector + args) into a single string. */
static void atoms_to_dsl_string(t_symbol *s, int argc, t_atom *argv,
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

/* Parse a DSL string into an AST. Returns NULL on parse error. */
static ASTNode *parse_dsl(const char *src) {
    if (!src) return NULL;
    Token *tokens = (Token *)getbytes(sizeof(Token) * 2048);
    if (!tokens) return NULL;
    int n_tok = tokenize(src, tokens, 2048);
    if (n_tok < 0) { freebytes(tokens, sizeof(Token) * 2048); return NULL; }
    ASTNode *root = parse(tokens, n_tok);
    freebytes(tokens, sizeof(Token) * 2048);
    return root;  /* NULL on parse failure — caller checks */
}

static void emit_expression(t_ns_seq_propose *x, const char *dsl) {
    if (!dsl || dsl[0] == '\0') return;
    outlet_anything(x->x_out, gensym(dsl), 0, NULL);
}

/* Push a new parent onto the 2-deep history. */
static void push_parent(t_ns_seq_propose *x, const char *dsl) {
    /* shift parent0 → parent1 */
    if (x->x_has_parent0) {
        memcpy(x->x_parent1, x->x_parent0, NS_PROPOSE_BUFSIZE);
        x->x_has_parent1 = 1;
    }
    snprintf(x->x_parent0, NS_PROPOSE_BUFSIZE, "%s", dsl);
    x->x_has_parent0 = 1;
}

/* Generate one offspring per current mode, with bounds-retry. Returns 1 on
 * success and writes the result into `out_dsl`. */
static int produce_offspring(t_ns_seq_propose *x, char *out_dsl, int max) {
    out_dsl[0] = '\0';
    for (int attempt = 0; attempt < 10; attempt++) {
        ASTNode *result = NULL;

        if (x->x_mode == NS_PROPOSE_RANDOM) {
            result = ns_gen_tree(&x->x_rng, NS_MAX_AST_DEPTH, 0);
        } else if (x->x_mode == NS_PROPOSE_CROSSOVER) {
            if (!x->x_has_parent0 || !x->x_has_parent1) return 0;
            ASTNode *a = parse_dsl(x->x_parent0);
            ASTNode *b = parse_dsl(x->x_parent1);
            if (a && b) result = ns_crossover(a, b, &x->x_rng);
            if (a) ast_free(a);
            if (b) ast_free(b);
        } else {
            /* mutate */
            if (!x->x_has_parent0) return 0;
            ASTNode *p = parse_dsl(x->x_parent0);
            if (p) {
                result = ns_mutate(&x->x_rng, p, x->x_rate, 0);
                ast_free(p);
            }
        }

        if (!result) continue;
        int sz = ns_ast_size(result);
        int dp = ns_ast_depth(result);
        if (sz <= x->x_max_size && dp <= x->x_max_depth) {
            int ok = ns_ast_render(result, out_dsl, max);
            ast_free(result);
            return ok;
        }
        ast_free(result);
    }
    return 0;
}

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_seq_propose_emit_from_cache(t_ns_seq_propose *x) {
    char buf[NS_PROPOSE_BUFSIZE];
    if (produce_offspring(x, buf, NS_PROPOSE_BUFSIZE)) {
        emit_expression(x, buf);
    }
}

static void ns_seq_propose_anything(t_ns_seq_propose *x, t_symbol *s, int argc, t_atom *argv) {
    char buf[NS_PROPOSE_BUFSIZE];
    atoms_to_dsl_string(s, argc, argv, buf, NS_PROPOSE_BUFSIZE);
    if (buf[0] == '\0') return;
    push_parent(x, buf);
    ns_seq_propose_emit_from_cache(x);
}

static void ns_seq_propose_list(t_ns_seq_propose *x, t_symbol *s, int argc, t_atom *argv) {
    /* In Pd a "list" message has no real selector — atoms_to_dsl_string skips
     * the selector for &s_list anyway. */
    (void)s;
    ns_seq_propose_anything(x, &s_list, argc, argv);
}

static void ns_seq_propose_symbol(t_ns_seq_propose *x, t_symbol *s) {
    push_parent(x, s->s_name);
    ns_seq_propose_emit_from_cache(x);
}

static void ns_seq_propose_bang(t_ns_seq_propose *x) {
    ns_seq_propose_emit_from_cache(x);
}

/* ======================================================================== */
/* PROXY (right inlet)                                                      */
/* ======================================================================== */

static void ns_seq_propose_proxy_rate(t_ns_seq_propose_proxy *p, t_floatarg f) {
    if (f < 0.0f) f = 0.0f; if (f > 1.0f) f = 1.0f;
    p->p_owner->x_rate = (float)f;
}

static void ns_seq_propose_proxy_mode(t_ns_seq_propose_proxy *p, t_symbol *s) {
    if (!s || !s->s_name) return;
    if (strcmp(s->s_name, "mutate") == 0)         p->p_owner->x_mode = NS_PROPOSE_MUTATE;
    else if (strcmp(s->s_name, "crossover") == 0) p->p_owner->x_mode = NS_PROPOSE_CROSSOVER;
    else if (strcmp(s->s_name, "random") == 0)    p->p_owner->x_mode = NS_PROPOSE_RANDOM;
    else pd_error(p->p_owner, "ns_seq_propose: unknown mode '%s'", s->s_name);
}

static void ns_seq_propose_proxy_seed(t_ns_seq_propose_proxy *p, t_floatarg f) {
    ns_rng_seed(&p->p_owner->x_rng, (uint64_t)(int64_t)f);
}

static void ns_seq_propose_proxy_max_size(t_ns_seq_propose_proxy *p, t_floatarg f) {
    int n = (int)f;
    if (n < 3) n = 3;
    p->p_owner->x_max_size = n;
}

static void ns_seq_propose_proxy_max_depth(t_ns_seq_propose_proxy *p, t_floatarg f) {
    int n = (int)f;
    if (n < 1) n = 1;
    p->p_owner->x_max_depth = n;
}

static void ns_seq_propose_proxy_clear(t_ns_seq_propose_proxy *p) {
    p->p_owner->x_has_parent0 = 0;
    p->p_owner->x_has_parent1 = 0;
}

/* Explicit-pair crossover: `crossover <expr_a> <expr_b>` on the right inlet
 * crosses two specific expressions and emits the offspring without
 * disturbing the 2-deep parent history. Designed to wire straight to
 * [ns_corpus]'s `pair` output: the corpus draws two parents from its pool
 * and emits them as `crossover <a> <b>`, this method handles the splice.
 *
 * Both args must be A_SYMBOL atoms carrying full DSL expressions. We use
 * the same bounds-retry policy as produce_offspring so the result respects
 * max_size / max_depth. */
static void ns_seq_propose_proxy_crossover(t_ns_seq_propose_proxy *p,
                                           t_symbol *sel, int argc, t_atom *argv) {
    (void)sel;
    t_ns_seq_propose *x = p->p_owner;
    if (argc < 2) {
        pd_error(x, "ns_seq_propose: crossover needs two expressions");
        return;
    }
    if (argv[0].a_type != A_SYMBOL || argv[1].a_type != A_SYMBOL) {
        pd_error(x, "ns_seq_propose: crossover args must be symbols");
        return;
    }
    const char *src_a = atom_getsymbol(&argv[0])->s_name;
    const char *src_b = atom_getsymbol(&argv[1])->s_name;
    if (!src_a || !src_b || src_a[0] == '\0' || src_b[0] == '\0') {
        pd_error(x, "ns_seq_propose: crossover args empty");
        return;
    }

    char out_dsl[NS_PROPOSE_BUFSIZE];
    out_dsl[0] = '\0';

    for (int attempt = 0; attempt < 10; attempt++) {
        ASTNode *a = parse_dsl(src_a);
        ASTNode *b = parse_dsl(src_b);
        ASTNode *result = NULL;
        if (a && b) result = ns_crossover(a, b, &x->x_rng);
        if (a) ast_free(a);
        if (b) ast_free(b);

        if (!result) continue;
        int sz = ns_ast_size(result);
        int dp = ns_ast_depth(result);
        if (sz <= x->x_max_size && dp <= x->x_max_depth) {
            int ok = ns_ast_render(result, out_dsl, NS_PROPOSE_BUFSIZE);
            ast_free(result);
            if (ok) {
                emit_expression(x, out_dsl);
                return;
            }
            return;
        }
        ast_free(result);
    }
    /* Fell through 10 attempts. Emit nothing — caller can detect via
     * the absence of an outlet message. */
}

/* Silent parent push: any list/anything message arriving on the right inlet
 * is treated as an expression to cache as parent — without triggering a new
 * proposal. This is what enables feedback loops: a downstream filter (e.g.
 * [ns_spigot] gating on novelty) can route admitted offspring back to the
 * right inlet so they become parents for the next bang. */
static void ns_seq_propose_proxy_anything(t_ns_seq_propose_proxy *p,
                                          t_symbol *s, int argc, t_atom *argv) {
    char buf[NS_PROPOSE_BUFSIZE];
    atoms_to_dsl_string(s, argc, argv, buf, NS_PROPOSE_BUFSIZE);
    if (buf[0] == '\0') return;
    push_parent(p->p_owner, buf);
}

static void ns_seq_propose_proxy_list(t_ns_seq_propose_proxy *p,
                                       t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    ns_seq_propose_proxy_anything(p, &s_list, argc, argv);
}

static void ns_seq_propose_proxy_symbol(t_ns_seq_propose_proxy *p, t_symbol *s) {
    if (!s || !s->s_name || s->s_name[0] == '\0') return;
    push_parent(p->p_owner, s->s_name);
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                 */
/* ======================================================================== */

static void *ns_seq_propose_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_seq_propose *x = (t_ns_seq_propose *)pd_new(ns_seq_propose_class);

    x->x_mode = NS_PROPOSE_MUTATE;
    x->x_rate = 0.2f;
    x->x_max_size = NS_MAX_AST_SIZE;
    x->x_max_depth = NS_MAX_AST_DEPTH;
    x->x_has_parent0 = 0;
    x->x_has_parent1 = 0;
    x->x_parent0[0] = '\0';
    x->x_parent1[0] = '\0';

    int ai = 0;
    if (argc > ai && argv[ai].a_type == A_SYMBOL) {
        const char *m = atom_getsymbol(&argv[ai])->s_name;
        if (strcmp(m, "mutate") == 0)         x->x_mode = NS_PROPOSE_MUTATE;
        else if (strcmp(m, "crossover") == 0) x->x_mode = NS_PROPOSE_CROSSOVER;
        else if (strcmp(m, "random") == 0)    x->x_mode = NS_PROPOSE_RANDOM;
        ai++;
    }
    if (argc > ai && argv[ai].a_type == A_FLOAT) {
        float r = (float)atom_getfloat(&argv[ai]);
        if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
        x->x_rate = r;
    }

    /* RNG seed from time + pointer for non-determinism by default. */
    uint64_t seed = (uint64_t)time(NULL);
    seed ^= ((uint64_t)(uintptr_t)x) * 0x9E3779B97F4A7C15ULL;
    ns_rng_seed(&x->x_rng, seed);

    x->x_proxy.p_pd = ns_seq_propose_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    x->x_out = outlet_new(&x->x_obj, &s_anything);
    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_seq_propose_setup(void) {
    ns_seq_propose_proxy_class = class_new(gensym("_ns_seq_propose_proxy"),
        0, 0, sizeof(t_ns_seq_propose_proxy), CLASS_PD, 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_rate,
                    gensym("rate"), A_FLOAT, 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_mode,
                    gensym("mode"), A_SYMBOL, 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_seed,
                    gensym("seed"), A_FLOAT, 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_max_size,
                    gensym("max_size"), A_FLOAT, 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_max_depth,
                    gensym("max_depth"), A_FLOAT, 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_clear,
                    gensym("clear"), 0);
    class_addmethod(ns_seq_propose_proxy_class, (t_method)ns_seq_propose_proxy_crossover,
                    gensym("crossover"), A_GIMME, 0);
    /* Silent-parent-push handlers: any message NOT matching a control selector
     * above is treated as an expression to cache without emitting. */
    class_addanything(ns_seq_propose_proxy_class, ns_seq_propose_proxy_anything);
    class_addlist(ns_seq_propose_proxy_class, ns_seq_propose_proxy_list);
    class_addsymbol(ns_seq_propose_proxy_class, ns_seq_propose_proxy_symbol);

    ns_seq_propose_class = class_new(gensym("ns_seq_propose"),
        (t_newmethod)ns_seq_propose_new,
        0,
        sizeof(t_ns_seq_propose),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addanything(ns_seq_propose_class, ns_seq_propose_anything);
    class_addlist(ns_seq_propose_class, ns_seq_propose_list);
    class_addsymbol(ns_seq_propose_class, ns_seq_propose_symbol);
    class_addbang(ns_seq_propose_class, ns_seq_propose_bang);

    post("ns_seq_propose %s - AST-level alien DSL evolution (mutate / crossover / random)",
         NS_VERSION_STRING);
}
