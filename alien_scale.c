/*
 * alien_scale - List-level scale quantizer for Pure Data
 *
 *   [alien_scale 0 2 4 5 7 9 11]
 *
 * Sits between [alien] and [else/sequencer]. Requantizes every note in
 * an incoming tape (list) to the nearest scale tone.  Rests ("-") pass
 * through unchanged.
 *
 * Left inlet:  list (tape from alien) — quantized and output immediately
 * Right inlet: list (new scale pitch classes, cold)
 * Messages:    root <n> — set root note
 *              bang     — re-output last quantized result
 */

#include "alien_core.h"

#define SCALE_MAX_STEPS 1024
#define SCALE_MAX_SCALE 12

static t_class *alien_scale_class;
static t_class *scale_proxy_class;

typedef struct _alien_scale t_alien_scale;

typedef struct _scale_proxy {
    t_pd p_pd;
    t_alien_scale *p_owner;
} t_scale_proxy;

struct _alien_scale {
    t_object x_obj;
    t_outlet *x_out;
    int x_scale[SCALE_MAX_SCALE];
    int x_scale_len;
    int x_root;
    t_atom x_last_output[SCALE_MAX_STEPS];
    int x_last_len;
    t_scale_proxy x_proxy;
};

/* ======================================================================== */
/* CORE PROCESSING                                                          */
/* ======================================================================== */

static void scale_process(t_alien_scale *x, int argc, t_atom *argv) {
    x->x_last_len = 0;
    for (int i = 0; i < argc && i < SCALE_MAX_STEPS; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int note = (int)atom_getfloat(&argv[i]);
            int quantized = alien_snap_to_scale(note, x->x_scale,
                                                x->x_scale_len, x->x_root);
            SETFLOAT(&x->x_last_output[x->x_last_len], (t_float)quantized);
        } else if (argv[i].a_type == A_SYMBOL) {
            SETSYMBOL(&x->x_last_output[x->x_last_len],
                      atom_getsymbol(&argv[i]));
        }
        x->x_last_len++;
    }
    if (x->x_last_len > 0)
        outlet_list(x->x_out, &s_list, x->x_last_len, x->x_last_output);
}

/* ======================================================================== */
/* LEFT INLET                                                               */
/* ======================================================================== */

static void scale_list(t_alien_scale *x, t_symbol *s,
                       int argc, t_atom *argv)
{
    (void)s;
    scale_process(x, argc, argv);
}

static void scale_anything(t_alien_scale *x, t_symbol *s,
                           int argc, t_atom *argv)
{
    /* Reconstruct: selector becomes first atom */
    t_atom buf[SCALE_MAX_STEPS];
    if (argc + 1 > SCALE_MAX_STEPS) return;
    SETSYMBOL(&buf[0], s);
    for (int i = 0; i < argc; i++) buf[i + 1] = argv[i];
    scale_process(x, argc + 1, buf);
}

static void scale_bang(t_alien_scale *x) {
    if (x->x_last_len > 0)
        outlet_list(x->x_out, &s_list, x->x_last_len, x->x_last_output);
}

/* ======================================================================== */
/* MESSAGES                                                                 */
/* ======================================================================== */

static void scale_root(t_alien_scale *x, t_floatarg f) {
    int r = (int)f;
    if (r < 0) r = 0;
    if (r > 127) r = 127;
    x->x_root = r;
}

/* ======================================================================== */
/* RIGHT INLET (proxy — cold)                                               */
/* ======================================================================== */

static void scale_proxy_list(t_scale_proxy *p, t_symbol *s,
                             int argc, t_atom *argv)
{
    (void)s;
    t_alien_scale *x = p->p_owner;
    x->x_scale_len = 0;
    for (int i = 0; i < argc && i < SCALE_MAX_SCALE; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int pc = (int)atom_getfloat(&argv[i]);
            if (pc >= 0 && pc <= 11) {
                x->x_scale[x->x_scale_len++] = pc;
            }
        }
    }
}

static void scale_proxy_anything(t_scale_proxy *p, t_symbol *s,
                                 int argc, t_atom *argv)
{
    (void)s;
    scale_proxy_list(p, &s_list, argc, argv);
}

/* ======================================================================== */
/* CONSTRUCTOR                                                              */
/* ======================================================================== */

static void *scale_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien_scale *x = (t_alien_scale *)pd_new(alien_scale_class);

    x->x_root = 0;
    x->x_scale_len = 0;
    x->x_last_len = 0;

    if (argc > 0) {
        for (int i = 0; i < argc && i < SCALE_MAX_SCALE; i++) {
            if (argv[i].a_type == A_FLOAT) {
                int pc = (int)atom_getfloat(&argv[i]);
                if (pc >= 0 && pc <= 11) {
                    x->x_scale[x->x_scale_len++] = pc;
                }
            }
        }
    } else {
        for (int i = 0; i < 12; i++) {
            x->x_scale[i] = i;
        }
        x->x_scale_len = 12;
    }

    /* Right inlet via proxy */
    x->x_proxy.p_pd = scale_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlet */
    x->x_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void alien_scale_setup(void) {
    scale_proxy_class = class_new(gensym("_alien_scale_proxy"),
        0, 0, sizeof(t_scale_proxy), CLASS_PD, 0);
    class_addlist(scale_proxy_class, scale_proxy_list);
    class_addanything(scale_proxy_class, scale_proxy_anything);

    alien_scale_class = class_new(gensym("alien_scale"),
        (t_newmethod)scale_new,
        0,
        sizeof(t_alien_scale),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addbang(alien_scale_class, scale_bang);
    class_addlist(alien_scale_class, scale_list);
    class_addanything(alien_scale_class, scale_anything);
    class_addmethod(alien_scale_class, (t_method)scale_root,
        gensym("root"), A_FLOAT, 0);

    post("alien_scale %s - list scale quantizer", ALIEN_VERSION_STRING);
}
