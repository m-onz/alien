/*
 * ns_seq_info - Report (size, depth) of an alien DSL expression for Pure Data
 *
 *   [ns_seq_info]
 *
 * Hot left inlet (anything / list / symbol):
 *     An alien DSL expression in either form:
 *       - atom-list: e.g. clicked from a [msg] box
 *       - single symbol: e.g. as emitted by [alien_wrap] or [ns_seq_propose]
 *
 * Outlet:
 *     left: list of 2 floats — (size, depth)
 *           where size is the count of AST nodes and depth is the maximum
 *           nesting depth (0 for a leaf, 1 for a single-level operator, etc.).
 *
 * Useful for tracking complexification over time: log the size column
 * alongside score in ns_log to see whether evolved offspring are growing.
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"
#include "ns_alien_ast.h"

#include <string.h>

/* ======================================================================== */

static t_class *ns_seq_info_class;

typedef struct _ns_seq_info {
    t_object x_obj;
    t_outlet *x_out;
} t_ns_seq_info;

#define NS_SEQ_INFO_BUFSIZE 4096

/* Same atom→string reassembler as ns_seq_propose. */
static void atoms_to_dsl(t_symbol *s, int argc, t_atom *argv, char *buf, int max) {
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

static void emit_info(t_ns_seq_info *x, int size, int depth) {
    t_atom out[2];
    SETFLOAT(&out[0], (t_float)size);
    SETFLOAT(&out[1], (t_float)depth);
    outlet_list(x->x_out, &s_list, 2, out);
}

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_seq_info_anything(t_ns_seq_info *x, t_symbol *s, int argc, t_atom *argv) {
    char buf[NS_SEQ_INFO_BUFSIZE];
    atoms_to_dsl(s, argc, argv, buf, NS_SEQ_INFO_BUFSIZE);
    if (buf[0] == '\0') {
        emit_info(x, 0, 0);
        return;
    }
    Token *tokens = (Token *)getbytes(sizeof(Token) * 2048);
    if (!tokens) {
        emit_info(x, 0, 0);
        return;
    }
    int n_tok = tokenize(buf, tokens, 2048);
    if (n_tok < 0) {
        freebytes(tokens, sizeof(Token) * 2048);
        emit_info(x, 0, 0);
        return;
    }
    ASTNode *root = parse(tokens, n_tok);
    freebytes(tokens, sizeof(Token) * 2048);
    if (!root) {
        emit_info(x, 0, 0);
        return;
    }
    int size = ns_ast_size(root);
    int depth = ns_ast_depth(root);
    ast_free(root);
    emit_info(x, size, depth);
}

static void ns_seq_info_list(t_ns_seq_info *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    ns_seq_info_anything(x, &s_list, argc, argv);
}

static void ns_seq_info_symbol(t_ns_seq_info *x, t_symbol *s) {
    if (!s) { emit_info(x, 0, 0); return; }
    ns_seq_info_anything(x, s, 0, NULL);
}

/* ======================================================================== */
/* CONSTRUCTOR / SETUP                                                      */
/* ======================================================================== */

static void *ns_seq_info_new(void) {
    t_ns_seq_info *x = (t_ns_seq_info *)pd_new(ns_seq_info_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}

void ns_seq_info_setup(void) {
    ns_seq_info_class = class_new(gensym("ns_seq_info"),
        (t_newmethod)ns_seq_info_new,
        0,
        sizeof(t_ns_seq_info),
        CLASS_DEFAULT,
        0);

    class_addanything(ns_seq_info_class, ns_seq_info_anything);
    class_addlist(ns_seq_info_class, ns_seq_info_list);
    class_addsymbol(ns_seq_info_class, ns_seq_info_symbol);

    post("ns_seq_info %s - alien DSL expr → (size, depth)", NS_VERSION_STRING);
}
