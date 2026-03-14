/*
 * alien_wrap - DSL expression wrapper for Pure Data
 *
 *   [alien_wrap (add _ 12)]
 *
 * Substitutes _ in the incoming message to allow wrapping patterns in a chainable manner
 * Chain multiple wraps to build up complex transformations, then feed the final output to [alien].
 *
 * Left inlet:  DSL expression — substituted into template, output sent
 * Right inlet: new template (cold — do/.es not trigger output)
 * Bang:        re-output with last input + current template
 */

#include "m_pd.h"
#include <string.h>
#include <stdio.h>

#define WRAP_BUFSIZE 4096
#define WRAP_OUTSIZE 8192

/* ======================================================================== */
/* FORWARD DECLARATIONS                                                     */
/* ======================================================================== */

static t_class *alien_wrap_class;
static t_class *wrap_proxy_class;

typedef struct _alien_wrap t_alien_wrap;

/* Proxy receives messages on the right inlet */
typedef struct _wrap_proxy {
    t_pd p_pd;
    t_alien_wrap *p_owner;
} t_wrap_proxy;

struct _alien_wrap {
    t_object x_obj;
    t_outlet *x_out;
    char x_template[WRAP_BUFSIZE];
    char x_last_input[WRAP_BUFSIZE];
    int x_has_input;
    t_wrap_proxy x_proxy;
};

/* ======================================================================== */
/* HELPERS                                                                  */
/* ======================================================================== */

/* Reconstruct Pd atoms into a plain string */
static void wrap_atoms_to_string(char *buf, int bufsize,
                                 t_symbol *s, int argc, t_atom *argv)
{
    char *p = buf;
    size_t remaining = (size_t)bufsize - 1;

    /* Include selector unless it is a Pd built-in */
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

/* A placeholder boundary is any character that cannot be part of a name */
static int wrap_is_boundary(char c) {
    return c == '\0' || c == ' ' || c == '(' || c == ')' ||
           c == '\t' || c == '\n';
}

/* Replace the first standalone _ in template with input */
static void wrap_substitute(const char *tmpl, const char *input,
                            char *out, int outsize)
{
    char *p = out;
    size_t remaining = (size_t)outsize - 1;
    const char *t = tmpl;
    size_t input_len = strlen(input);
    int replaced = 0;

    while (*t && remaining > 0) {
        if (!replaced && *t == '_' &&
            (t == tmpl || wrap_is_boundary(*(t - 1))) &&
            wrap_is_boundary(*(t + 1)))
        {
            if (input_len < remaining) {
                memcpy(p, input, input_len);
                p += input_len;
                remaining -= input_len;
            }
            t++;
            replaced = 1;
        } else {
            *p++ = *t++;
            remaining--;
        }
    }
    *p = '\0';
}

/* Perform substitution and send the result out */
static void wrap_do_output(t_alien_wrap *x) {
    if (!x->x_has_input || x->x_template[0] == '\0') return;

    char output[WRAP_OUTSIZE];
    wrap_substitute(x->x_template, x->x_last_input,
                    output, WRAP_OUTSIZE);

    if (output[0] != '\0')
        outlet_anything(x->x_out, gensym(output), 0, NULL);
}

/* ======================================================================== */
/* LEFT INLET                                                               */
/* ======================================================================== */

static void wrap_anything(t_alien_wrap *x, t_symbol *s,
                          int argc, t_atom *argv)
{
    wrap_atoms_to_string(x->x_last_input, WRAP_BUFSIZE, s, argc, argv);
    x->x_has_input = 1;
    wrap_do_output(x);
}

static void wrap_list(t_alien_wrap *x, t_symbol *s,
                      int argc, t_atom *argv)
{
    (void)s;
    if (argc > 0 && argv[0].a_type == A_SYMBOL)
        wrap_anything(x, atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
    else {
        wrap_atoms_to_string(x->x_last_input, WRAP_BUFSIZE,
                             &s_list, argc, argv);
        x->x_has_input = 1;
        wrap_do_output(x);
    }
}

static void wrap_symbol(t_alien_wrap *x, t_symbol *s) {
    snprintf(x->x_last_input, WRAP_BUFSIZE, "%s", s->s_name);
    x->x_has_input = 1;
    wrap_do_output(x);
}

static void wrap_float(t_alien_wrap *x, t_floatarg f) {
    if (f == (int)f)
        snprintf(x->x_last_input, WRAP_BUFSIZE, "%d", (int)f);
    else
        snprintf(x->x_last_input, WRAP_BUFSIZE, "%g", f);
    x->x_has_input = 1;
    wrap_do_output(x);
}

static void wrap_bang(t_alien_wrap *x) {
    wrap_do_output(x);
}

/* ======================================================================== */
/* RIGHT INLET (proxy — cold)                                               */
/* ======================================================================== */

static void wrap_proxy_anything(t_wrap_proxy *p, t_symbol *s,
                                int argc, t_atom *argv)
{
    wrap_atoms_to_string(p->p_owner->x_template, WRAP_BUFSIZE,
                         s, argc, argv);
}

static void wrap_proxy_list(t_wrap_proxy *p, t_symbol *s,
                            int argc, t_atom *argv)
{
    (void)s;
    if (argc > 0 && argv[0].a_type == A_SYMBOL)
        wrap_proxy_anything(p, atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
    else
        wrap_atoms_to_string(p->p_owner->x_template, WRAP_BUFSIZE,
                             &s_list, argc, argv);
}

static void wrap_proxy_symbol(t_wrap_proxy *p, t_symbol *s) {
    snprintf(p->p_owner->x_template, WRAP_BUFSIZE, "%s", s->s_name);
}

/* ======================================================================== */
/* CONSTRUCTOR                                                              */
/* ======================================================================== */

static void *wrap_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien_wrap *x = (t_alien_wrap *)pd_new(alien_wrap_class);

    x->x_template[0] = '\0';
    x->x_last_input[0] = '\0';
    x->x_has_input = 0;

    /* Creation args are the template */
    if (argc > 0)
        wrap_atoms_to_string(x->x_template, WRAP_BUFSIZE, NULL, argc, argv);

    /* Right inlet via proxy */
    x->x_proxy.p_pd = wrap_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlet */
    x->x_out = outlet_new(&x->x_obj, &s_anything);

    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void alien_wrap_setup(void) {
    /* Proxy class for right inlet routing */
    wrap_proxy_class = class_new(gensym("_alien_wrap_proxy"),
        0, 0, sizeof(t_wrap_proxy), CLASS_PD, 0);
    class_addanything(wrap_proxy_class, wrap_proxy_anything);
    class_addlist(wrap_proxy_class, wrap_proxy_list);
    class_addsymbol(wrap_proxy_class, wrap_proxy_symbol);

    /* Main class */
    alien_wrap_class = class_new(gensym("alien_wrap"),
        (t_newmethod)wrap_new,
        0,
        sizeof(t_alien_wrap),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addbang(alien_wrap_class, wrap_bang);
    class_addfloat(alien_wrap_class, wrap_float);
    class_addsymbol(alien_wrap_class, wrap_symbol);
    class_addlist(alien_wrap_class, wrap_list);
    class_addanything(alien_wrap_class, wrap_anything);
}
