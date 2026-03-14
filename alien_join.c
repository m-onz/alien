/*
 * alien_join - Multi-inlet pattern joiner for Pure Data
 *
 *   [alien_join 3]
 *
 * Joins N inlet expressions into (seq <input_1> <input_2> ... <input_N>).
 * Any new message on any inlet triggers output of the joined sequence.
 * Allows building long patterns from manageable sections.
 *
 * Creation arg: number of inlets (default 2)
 * All inlets:   DSL expression (symbol/list/anything)
 * Bang:         re-output current joined result
 * Outlet:       joined expression as symbol message
 */

#include "m_pd.h"
#include <string.h>
#include <stdio.h>

#define JOIN_BUFSIZE 4096
#define JOIN_OUTSIZE 16384
#define JOIN_MAX_INLETS 32

static t_class *alien_join_class;
static t_class *join_proxy_class;

typedef struct _alien_join t_alien_join;

typedef struct _join_proxy {
    t_pd p_pd;
    t_alien_join *p_owner;
    int p_index;
} t_join_proxy;

struct _alien_join {
    t_object x_obj;
    t_outlet *x_out;
    int x_n;                               /* number of inlets */
    char x_inputs[JOIN_MAX_INLETS][JOIN_BUFSIZE];
    int x_has_input[JOIN_MAX_INLETS];
    t_join_proxy x_proxies[JOIN_MAX_INLETS]; /* [0] unused, proxies for 1..n-1 */
};

/* ======================================================================== */
/* HELPERS                                                                  */
/* ======================================================================== */

static void join_atoms_to_string(char *buf, int bufsize,
                                 t_symbol *s, int argc, t_atom *argv)
{
    char *p = buf;
    size_t remaining = (size_t)bufsize - 1;

    if (s && s != &s_list && s != &s_symbol && s != &s_float && s != &s_bang) {
        int len = snprintf(p, remaining, "%s", s->s_name);
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }
        if (argc > 0 && remaining > 0) { *p++ = ' '; remaining--; }
    }

    for (int i = 0; i < argc && remaining > 1; i++) {
        int len = 0;
        if (argv[i].a_type == A_FLOAT) {
            float f = atom_getfloat(&argv[i]);
            if (f == (int)f)
                len = snprintf(p, remaining, "%s%d", (i > 0 ? " " : ""), (int)f);
            else
                len = snprintf(p, remaining, "%s%g", (i > 0 ? " " : ""), f);
        } else if (argv[i].a_type == A_SYMBOL) {
            len = snprintf(p, remaining, "%s%s",
                           (i > 0 ? " " : ""),
                           atom_getsymbol(&argv[i])->s_name);
        }
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }
    }
    *p = '\0';
}

static void join_do_output(t_alien_join *x) {
    char output[JOIN_OUTSIZE];
    char *p = output;
    size_t remaining = (size_t)JOIN_OUTSIZE - 1;

    int len = snprintf(p, remaining, "(seq");
    if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }

    for (int i = 0; i < x->x_n; i++) {
        if (x->x_has_input[i] && x->x_inputs[i][0] != '\0') {
            len = snprintf(p, remaining, " %s", x->x_inputs[i]);
            if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }
        }
    }

    len = snprintf(p, remaining, ")");
    if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }

    *p = '\0';

    if (output[0] != '\0')
        outlet_anything(x->x_out, gensym(output), 0, NULL);
}

static void join_store(t_alien_join *x, int index,
                       t_symbol *s, int argc, t_atom *argv)
{
    if (index < 0 || index >= x->x_n) return;
    join_atoms_to_string(x->x_inputs[index], JOIN_BUFSIZE, s, argc, argv);
    x->x_has_input[index] = 1;
    join_do_output(x);
}

/* ======================================================================== */
/* LEFT INLET (index 0)                                                     */
/* ======================================================================== */

static void join_anything(t_alien_join *x, t_symbol *s,
                          int argc, t_atom *argv)
{
    join_store(x, 0, s, argc, argv);
}

static void join_list(t_alien_join *x, t_symbol *s,
                      int argc, t_atom *argv)
{
    (void)s;
    if (argc > 0 && argv[0].a_type == A_SYMBOL)
        join_anything(x, atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
    else
        join_store(x, 0, &s_list, argc, argv);
}

static void join_symbol(t_alien_join *x, t_symbol *s) {
    snprintf(x->x_inputs[0], JOIN_BUFSIZE, "%s", s->s_name);
    x->x_has_input[0] = 1;
    join_do_output(x);
}

static void join_float(t_alien_join *x, t_floatarg f) {
    if (f == (int)f)
        snprintf(x->x_inputs[0], JOIN_BUFSIZE, "%d", (int)f);
    else
        snprintf(x->x_inputs[0], JOIN_BUFSIZE, "%g", f);
    x->x_has_input[0] = 1;
    join_do_output(x);
}

static void join_bang(t_alien_join *x) {
    join_do_output(x);
}

/* ======================================================================== */
/* PROXY INLETS (index 1..n-1)                                              */
/* ======================================================================== */

static void join_proxy_anything(t_join_proxy *p, t_symbol *s,
                                int argc, t_atom *argv)
{
    join_store(p->p_owner, p->p_index, s, argc, argv);
}

static void join_proxy_list(t_join_proxy *p, t_symbol *s,
                            int argc, t_atom *argv)
{
    (void)s;
    if (argc > 0 && argv[0].a_type == A_SYMBOL)
        join_proxy_anything(p, atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
    else
        join_store(p->p_owner, p->p_index, &s_list, argc, argv);
}

static void join_proxy_symbol(t_join_proxy *p, t_symbol *s) {
    snprintf(p->p_owner->x_inputs[p->p_index], JOIN_BUFSIZE, "%s", s->s_name);
    p->p_owner->x_has_input[p->p_index] = 1;
    join_do_output(p->p_owner);
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                 */
/* ======================================================================== */

static void *join_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien_join *x = (t_alien_join *)pd_new(alien_join_class);

    x->x_n = 2;
    if (argc > 0 && argv[0].a_type == A_FLOAT) {
        int n = (int)atom_getfloat(&argv[0]);
        if (n < 2) n = 2;
        if (n > JOIN_MAX_INLETS) n = JOIN_MAX_INLETS;
        x->x_n = n;
    }

    for (int i = 0; i < x->x_n; i++) {
        x->x_inputs[i][0] = '\0';
        x->x_has_input[i] = 0;
    }

    /* Create proxy inlets for 1..n-1 */
    for (int i = 1; i < x->x_n; i++) {
        x->x_proxies[i].p_pd = join_proxy_class;
        x->x_proxies[i].p_owner = x;
        x->x_proxies[i].p_index = i;
        inlet_new(&x->x_obj, &x->x_proxies[i].p_pd, 0, 0);
    }

    x->x_out = outlet_new(&x->x_obj, &s_anything);

    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void alien_join_setup(void) {
    join_proxy_class = class_new(gensym("_alien_join_proxy"),
        0, 0, sizeof(t_join_proxy), CLASS_PD, 0);
    class_addanything(join_proxy_class, join_proxy_anything);
    class_addlist(join_proxy_class, join_proxy_list);
    class_addsymbol(join_proxy_class, join_proxy_symbol);

    alien_join_class = class_new(gensym("alien_join"),
        (t_newmethod)join_new,
        0,
        sizeof(t_alien_join),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addbang(alien_join_class, join_bang);
    class_addfloat(alien_join_class, join_float);
    class_addsymbol(alien_join_class, join_symbol);
    class_addlist(alien_join_class, join_list);
    class_addanything(alien_join_class, join_anything);

    post("alien_join - multi-inlet pattern joiner");
}
