/*
 * alien_snap - Scale quantizer for Pure Data
 *
 *   [alien_snap 0 2 4 5 7 9 11]
 *
 * Snaps incoming MIDI notes to the nearest scale tone.
 * Left inlet:  float (MIDI note) — quantized and output immediately
 * Right inlet: list (new scale pitch classes, cold)
 * Messages:    root <n> — set root note
 */

#include "alien_core.h"

#define SNAP_MAX_SCALE 12

static t_class *alien_snap_class;
static t_class *snap_proxy_class;

typedef struct _alien_snap t_alien_snap;

typedef struct _snap_proxy {
    t_pd p_pd;
    t_alien_snap *p_owner;
} t_snap_proxy;

struct _alien_snap {
    t_object x_obj;
    t_outlet *x_out;
    int x_scale[SNAP_MAX_SCALE];
    int x_scale_len;
    int x_root;
    t_snap_proxy x_proxy;
};

/* ======================================================================== */
/* LEFT INLET                                                               */
/* ======================================================================== */

static void snap_float(t_alien_snap *x, t_floatarg f) {
    int note = (int)f;
    int quantized = alien_snap_to_scale(note, x->x_scale, x->x_scale_len, x->x_root);
    outlet_float(x->x_out, (t_float)quantized);
}

/* ======================================================================== */
/* MESSAGES                                                                 */
/* ======================================================================== */

static void snap_root(t_alien_snap *x, t_floatarg f) {
    int r = (int)f;
    if (r < 0) r = 0;
    if (r > 127) r = 127;
    x->x_root = r;
}

/* ======================================================================== */
/* RIGHT INLET (proxy — cold)                                               */
/* ======================================================================== */

static void snap_proxy_list(t_snap_proxy *p, t_symbol *s,
                            int argc, t_atom *argv)
{
    (void)s;
    t_alien_snap *x = p->p_owner;
    x->x_scale_len = 0;
    for (int i = 0; i < argc && i < SNAP_MAX_SCALE; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int pc = (int)atom_getfloat(&argv[i]);
            if (pc >= 0 && pc <= 11) {
                x->x_scale[x->x_scale_len++] = pc;
            }
        }
    }
}

static void snap_proxy_anything(t_snap_proxy *p, t_symbol *s,
                                int argc, t_atom *argv)
{
    (void)s;
    snap_proxy_list(p, &s_list, argc, argv);
}

/* ======================================================================== */
/* CONSTRUCTOR                                                              */
/* ======================================================================== */

static void *snap_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien_snap *x = (t_alien_snap *)pd_new(alien_snap_class);

    x->x_root = 0;
    x->x_scale_len = 0;

    if (argc > 0) {
        for (int i = 0; i < argc && i < SNAP_MAX_SCALE; i++) {
            if (argv[i].a_type == A_FLOAT) {
                int pc = (int)atom_getfloat(&argv[i]);
                if (pc >= 0 && pc <= 11) {
                    x->x_scale[x->x_scale_len++] = pc;
                }
            }
        }
    } else {
        /* Default: chromatic (all 12 pitch classes) */
        for (int i = 0; i < 12; i++) {
            x->x_scale[i] = i;
        }
        x->x_scale_len = 12;
    }

    /* Right inlet via proxy */
    x->x_proxy.p_pd = snap_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlet */
    x->x_out = outlet_new(&x->x_obj, &s_float);

    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void alien_snap_setup(void) {
    snap_proxy_class = class_new(gensym("_alien_snap_proxy"),
        0, 0, sizeof(t_snap_proxy), CLASS_PD, 0);
    class_addlist(snap_proxy_class, snap_proxy_list);
    class_addanything(snap_proxy_class, snap_proxy_anything);

    alien_snap_class = class_new(gensym("alien_snap"),
        (t_newmethod)snap_new,
        0,
        sizeof(t_alien_snap),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addfloat(alien_snap_class, snap_float);
    class_addmethod(alien_snap_class, (t_method)snap_root,
        gensym("root"), A_FLOAT, 0);

    post("alien_snap %s - scale quantizer", ALIEN_VERSION_STRING);
}
