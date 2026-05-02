/*
 * ns_mutate - Perturb a list (Gaussian noise or bit-flip) for Pure Data
 *
 *   [ns_mutate]                  default: gaussian, sigma = 0.1
 *   [ns_mutate gaussian 0.1]     continuous, additive N(0, sigma)
 *   [ns_mutate bitflip 0.05]     binary; each cell flipped with prob = 0.05
 *
 * Hot left inlet (list):
 *     A vector to perturb. Outputs the mutated vector.
 *     Subsequent bangs re-mutate the most recently received list (with
 *     fresh randomness).
 *
 * Cold middle inlet (float):
 *     New mutation rate (sigma for gaussian, prob for bitflip).
 *
 * Cold right inlet (proxy):
 *     mode <symbol>          gaussian | bitflip
 *     seed <int>             reseed the RNG (deterministic from here)
 *     clip <float> <float>   clip output to [lo, hi]; clip none unsets
 *     bang                   re-mutate last input
 *
 * Outlet:
 *     left:  list            perturbed vector
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>
#include <stdio.h>
#include <time.h>

/* ======================================================================== */

static t_class *ns_mutate_class;
static t_class *ns_mutate_proxy_class;

typedef struct _ns_mutate t_ns_mutate;

typedef struct _ns_mutate_proxy {
    t_pd p_pd;
    t_ns_mutate *p_owner;
} t_ns_mutate_proxy;

typedef enum {
    NS_MODE_GAUSSIAN = 0,
    NS_MODE_BITFLIP = 1,
} ns_mutate_mode_t;

#define NS_MUTATE_STACK 256

struct _ns_mutate {
    t_object x_obj;
    t_outlet *x_out;
    ns_mutate_mode_t x_mode;
    float x_rate;
    int x_clip_enabled;
    float x_clip_lo, x_clip_hi;
    ns_rng_t x_rng;
    /* Last input cache for bang re-mutate. */
    float *x_last;
    int x_last_dim;
    int x_last_capacity;
    t_ns_mutate_proxy x_proxy;
};

/* ======================================================================== */

static void ns_mutate_emit(t_ns_mutate *x, const float *vec, int dim) {
    t_atom stack_atoms[NS_MUTATE_STACK];
    t_atom *atoms = stack_atoms;
    int allocated = 0;
    if (dim > NS_MUTATE_STACK) {
        atoms = (t_atom *)getbytes(sizeof(t_atom) * dim);
        if (!atoms) { pd_error(x, "ns_mutate: out of memory"); return; }
        allocated = 1;
    }
    for (int i = 0; i < dim; i++) SETFLOAT(&atoms[i], (t_float)vec[i]);
    outlet_list(x->x_out, &s_list, dim, atoms);
    if (allocated) freebytes(atoms, sizeof(t_atom) * dim);
}

/* Cache the input list into x_last so bang can re-mutate. */
static int cache_input(t_ns_mutate *x, int argc, t_atom *argv) {
    /* Validate. */
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "ns_mutate: list must be all floats");
            return 0;
        }
    }
    /* Resize cache if needed. */
    if (argc > x->x_last_capacity) {
        if (x->x_last) {
            x->x_last = (float *)resizebytes(x->x_last,
                sizeof(float) * x->x_last_capacity,
                sizeof(float) * argc);
        } else {
            x->x_last = (float *)getbytes(sizeof(float) * argc);
        }
        if (!x->x_last) {
            pd_error(x, "ns_mutate: out of memory");
            x->x_last_capacity = 0;
            return 0;
        }
        x->x_last_capacity = argc;
    }
    for (int i = 0; i < argc; i++) {
        x->x_last[i] = (float)atom_getfloat(&argv[i]);
    }
    x->x_last_dim = argc;
    return 1;
}

static void ns_mutate_apply(t_ns_mutate *x, float *vec, int dim) {
    if (x->x_mode == NS_MODE_GAUSSIAN) {
        ns_mutate_gaussian(vec, dim, x->x_rate, &x->x_rng);
    } else {
        ns_mutate_bitflip(vec, dim, x->x_rate, &x->x_rng);
    }
    if (x->x_clip_enabled) {
        ns_clip(vec, dim, x->x_clip_lo, x->x_clip_hi);
    }
}

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_mutate_list(t_ns_mutate *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc < 1) return;
    if (!cache_input(x, argc, argv)) return;

    /* Mutate a copy (we keep x_last as the original for bang re-mutate). */
    float stack_buf[NS_MUTATE_STACK];
    float *buf = stack_buf;
    int allocated = 0;
    if (argc > NS_MUTATE_STACK) {
        buf = (float *)getbytes(sizeof(float) * argc);
        if (!buf) { pd_error(x, "ns_mutate: out of memory"); return; }
        allocated = 1;
    }
    memcpy(buf, x->x_last, sizeof(float) * argc);
    ns_mutate_apply(x, buf, argc);
    ns_mutate_emit(x, buf, argc);
    if (allocated) freebytes(buf, sizeof(float) * argc);
}

static void ns_mutate_bang(t_ns_mutate *x) {
    if (x->x_last_dim <= 0) return;
    int dim = x->x_last_dim;
    float stack_buf[NS_MUTATE_STACK];
    float *buf = stack_buf;
    int allocated = 0;
    if (dim > NS_MUTATE_STACK) {
        buf = (float *)getbytes(sizeof(float) * dim);
        if (!buf) { pd_error(x, "ns_mutate: out of memory"); return; }
        allocated = 1;
    }
    memcpy(buf, x->x_last, sizeof(float) * dim);
    ns_mutate_apply(x, buf, dim);
    ns_mutate_emit(x, buf, dim);
    if (allocated) freebytes(buf, sizeof(float) * dim);
}

static void ns_mutate_float(t_ns_mutate *x, t_floatarg f) {
    /* Cold middle inlet would be cleaner, but using the float method on the
     * left inlet for the rate is non-idiomatic. Instead, expose the rate as
     * a `rate` message on the proxy. We treat a float on the hot inlet as
     * a single-element list. */
    t_atom a;
    SETFLOAT(&a, f);
    ns_mutate_list(x, &s_list, 1, &a);
}

/* ======================================================================== */
/* PROXY (right inlet) — control                                            */
/* ======================================================================== */

static void ns_mutate_proxy_mode(t_ns_mutate_proxy *p, t_symbol *s) {
    if (!s || !s->s_name) return;
    if (strcmp(s->s_name, "gaussian") == 0)     p->p_owner->x_mode = NS_MODE_GAUSSIAN;
    else if (strcmp(s->s_name, "bitflip") == 0) p->p_owner->x_mode = NS_MODE_BITFLIP;
    else pd_error(p->p_owner, "ns_mutate: unknown mode '%s' (use gaussian|bitflip)", s->s_name);
}

static void ns_mutate_proxy_rate(t_ns_mutate_proxy *p, t_floatarg f) {
    if (f < 0.0f) f = 0.0f;
    p->p_owner->x_rate = (float)f;
}

static void ns_mutate_proxy_seed(t_ns_mutate_proxy *p, t_floatarg f) {
    ns_rng_seed(&p->p_owner->x_rng, (uint64_t)(int64_t)f);
}

static void ns_mutate_proxy_clip(t_ns_mutate_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_mutate *x = p->p_owner;
    if (argc == 0) {
        x->x_clip_enabled = 0;
        return;
    }
    if (argc != 2 || argv[0].a_type != A_FLOAT || argv[1].a_type != A_FLOAT) {
        pd_error(x, "ns_mutate: clip needs two floats (or no args to disable)");
        return;
    }
    float lo = (float)atom_getfloat(&argv[0]);
    float hi = (float)atom_getfloat(&argv[1]);
    if (hi < lo) { float t = lo; lo = hi; hi = t; }
    x->x_clip_lo = lo;
    x->x_clip_hi = hi;
    x->x_clip_enabled = 1;
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                 */
/* ======================================================================== */

static void *ns_mutate_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_mutate *x = (t_ns_mutate *)pd_new(ns_mutate_class);

    /* Defaults */
    x->x_mode = NS_MODE_GAUSSIAN;
    x->x_rate = 0.1f;
    x->x_clip_enabled = 0;
    x->x_clip_lo = 0.0f;
    x->x_clip_hi = 1.0f;
    x->x_last = NULL;
    x->x_last_dim = 0;
    x->x_last_capacity = 0;

    /* Optional creation args: mode, rate */
    int ai = 0;
    if (argc > ai && argv[ai].a_type == A_SYMBOL) {
        const char *name = atom_getsymbol(&argv[ai])->s_name;
        if (strcmp(name, "gaussian") == 0)     x->x_mode = NS_MODE_GAUSSIAN;
        else if (strcmp(name, "bitflip") == 0) x->x_mode = NS_MODE_BITFLIP;
        ai++;
    }
    if (argc > ai && argv[ai].a_type == A_FLOAT) {
        float r = (float)atom_getfloat(&argv[ai]);
        if (r < 0.0f) r = 0.0f;
        x->x_rate = r;
        ai++;
    }

    /* Seed RNG from time + pointer for non-determinism by default. */
    uint64_t seed = (uint64_t)time(NULL);
    seed ^= ((uint64_t)(uintptr_t)x) * 0x9E3779B97F4A7C15ULL;
    ns_rng_seed(&x->x_rng, seed);

    /* Proxy right inlet for control. */
    x->x_proxy.p_pd = ns_mutate_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlet: list */
    x->x_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

static void ns_mutate_free(t_ns_mutate *x) {
    if (x->x_last) {
        freebytes(x->x_last, sizeof(float) * x->x_last_capacity);
    }
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_mutate_setup(void) {
    ns_mutate_proxy_class = class_new(gensym("_ns_mutate_proxy"),
        0, 0, sizeof(t_ns_mutate_proxy), CLASS_PD, 0);
    class_addmethod(ns_mutate_proxy_class, (t_method)ns_mutate_proxy_mode,
                    gensym("mode"), A_SYMBOL, 0);
    class_addmethod(ns_mutate_proxy_class, (t_method)ns_mutate_proxy_rate,
                    gensym("rate"), A_FLOAT, 0);
    class_addmethod(ns_mutate_proxy_class, (t_method)ns_mutate_proxy_seed,
                    gensym("seed"), A_FLOAT, 0);
    class_addmethod(ns_mutate_proxy_class, (t_method)ns_mutate_proxy_clip,
                    gensym("clip"), A_GIMME, 0);

    ns_mutate_class = class_new(gensym("ns_mutate"),
        (t_newmethod)ns_mutate_new,
        (t_method)ns_mutate_free,
        sizeof(t_ns_mutate),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addlist(ns_mutate_class, ns_mutate_list);
    class_addbang(ns_mutate_class, ns_mutate_bang);
    class_addfloat(ns_mutate_class, ns_mutate_float);

    post("ns_mutate %s - list perturbation (gaussian | bitflip)", NS_VERSION_STRING);
}
