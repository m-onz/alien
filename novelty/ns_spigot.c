/*
 * ns_spigot - Threshold-gated list passer for Pure Data
 *
 *   [ns_spigot 0.3]            pass stored list if incoming score >= 0.3
 *   [ns_spigot 0.3 above]      explicit "above" (same as default)
 *   [ns_spigot 0.3 below]      pass if incoming score <= 0.3
 *
 * Use:
 *     The right (cold) inlet stores the payload. The left (hot) inlet
 *     takes a score and triggers a decision: if the score meets the
 *     threshold, the stored payload is emitted on the left outlet.
 *     Otherwise the right outlet bangs (so a patch can count rejections).
 *
 *     Typical wiring around ns_archive:
 *
 *         [list]                      ; candidate
 *           |
 *         [t a a]                     ; right outlet fires first
 *           |   \
 *           |    +→ [ns_spigot 0.3] right inlet  (cache the list)
 *           |
 *           +→ [ns_archive shared]    ; produces score
 *               |
 *             score → [ns_spigot 0.3] left inlet  (trigger decision)
 *
 * Hot left inlet (float):
 *     Score. Triggers the decision. If a list has been stored and the
 *     score satisfies the relation, the list is emitted; otherwise the
 *     right outlet bangs.
 *
 * Cold right inlet (proxy):
 *     <list>            store payload (replaces previous)
 *     threshold <f>     change threshold
 *     mode <above|below>  flip relation; "above" = pass when score ≥ thr
 *     bang              re-evaluate with the last (score, list) pair
 *
 * Outlets:
 *     left:  list       passed payload
 *     right: bang       blocked
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>
#include <stdio.h>

/* ======================================================================== */

static t_class *ns_spigot_class;
static t_class *ns_spigot_proxy_class;

typedef struct _ns_spigot t_ns_spigot;

typedef struct _ns_spigot_proxy {
    t_pd p_pd;
    t_ns_spigot *p_owner;
} t_ns_spigot_proxy;

typedef enum {
    NS_SPIGOT_ABOVE = 0,    /* pass if score >= threshold */
    NS_SPIGOT_BELOW = 1,    /* pass if score <= threshold */
} ns_spigot_mode_t;

#define NS_SPIGOT_STACK 256

struct _ns_spigot {
    t_object x_obj;
    t_outlet *x_pass_out;     /* anything outlet — preserves message selector */
    t_outlet *x_block_out;    /* bang outlet */
    float x_threshold;
    ns_spigot_mode_t x_mode;
    /* Last received score (for bang re-eval). */
    float x_last_score;
    int x_has_score;
    /* Cached payload — selector + args, so single-symbol messages from
     * outlet_anything (e.g. ns_seq_propose's output) round-trip correctly. */
    t_symbol *x_payload_sel;
    t_atom *x_payload;
    int x_payload_len;
    int x_payload_cap;
    int x_has_payload;
    t_ns_spigot_proxy x_proxy;
};

/* ======================================================================== */
/* PAYLOAD CACHE                                                            */
/* ======================================================================== */

static int payload_store(t_ns_spigot *x, t_symbol *sel, int argc, t_atom *argv) {
    if (argc > x->x_payload_cap) {
        if (x->x_payload) {
            x->x_payload = (t_atom *)resizebytes(x->x_payload,
                sizeof(t_atom) * x->x_payload_cap,
                sizeof(t_atom) * argc);
        } else {
            x->x_payload = (t_atom *)getbytes(sizeof(t_atom) * argc);
        }
        if (!x->x_payload) {
            x->x_payload_cap = 0;
            x->x_has_payload = 0;
            pd_error(x, "ns_spigot: out of memory");
            return 0;
        }
        x->x_payload_cap = argc;
    }
    for (int i = 0; i < argc; i++) x->x_payload[i] = argv[i];
    x->x_payload_len = argc;
    x->x_payload_sel = sel ? sel : &s_list;
    x->x_has_payload = 1;
    return 1;
}

/* ======================================================================== */
/* DECISION                                                                 */
/* ======================================================================== */

static int decision(t_ns_spigot *x, float score) {
    if (x->x_mode == NS_SPIGOT_BELOW) {
        return (score <= x->x_threshold);
    }
    return (score >= x->x_threshold);
}

static void evaluate(t_ns_spigot *x) {
    if (!x->x_has_payload) {
        /* Score arrived but nothing to gate. Bang the block outlet so the
         * patch is aware. (Common during the very first iteration.) */
        outlet_bang(x->x_block_out);
        return;
    }
    if (decision(x, x->x_last_score)) {
        /* Re-emit with the original selector so single-symbol messages
         * (from outlet_anything) round-trip without losing data. */
        outlet_anything(x->x_pass_out, x->x_payload_sel,
                        x->x_payload_len, x->x_payload);
    } else {
        outlet_bang(x->x_block_out);
    }
}

/* ======================================================================== */
/* HOT LEFT INLET — float (score) triggers                                  */
/* ======================================================================== */

static void ns_spigot_float(t_ns_spigot *x, t_floatarg f) {
    x->x_last_score = (float)f;
    x->x_has_score = 1;
    evaluate(x);
}

/* ======================================================================== */
/* PROXY (right inlet) — list stores payload, messages configure            */
/* ======================================================================== */

static void ns_spigot_proxy_list(t_ns_spigot_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    payload_store(p->p_owner, s, argc, argv);
}

static void ns_spigot_proxy_anything(t_ns_spigot_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    /* Capture the selector too, so single-symbol messages from outlet_anything
     * (e.g. ns_seq_propose carrying the whole DSL string as the selector)
     * round-trip correctly through the gate. */
    payload_store(p->p_owner, s, argc, argv);
}

static void ns_spigot_proxy_threshold(t_ns_spigot_proxy *p, t_floatarg f) {
    p->p_owner->x_threshold = (float)f;
}

static void ns_spigot_proxy_mode(t_ns_spigot_proxy *p, t_symbol *s) {
    if (!s || !s->s_name) return;
    if (strcmp(s->s_name, "above") == 0)      p->p_owner->x_mode = NS_SPIGOT_ABOVE;
    else if (strcmp(s->s_name, "below") == 0) p->p_owner->x_mode = NS_SPIGOT_BELOW;
    else pd_error(p->p_owner, "ns_spigot: unknown mode '%s' (use above|below)", s->s_name);
}

static void ns_spigot_proxy_bang(t_ns_spigot_proxy *p) {
    /* Re-evaluate with the last known score. */
    if (!p->p_owner->x_has_score) return;
    evaluate(p->p_owner);
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                 */
/* ======================================================================== */

static void *ns_spigot_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_spigot *x = (t_ns_spigot *)pd_new(ns_spigot_class);

    /* Defaults */
    x->x_threshold = 0.0f;
    x->x_mode = NS_SPIGOT_ABOVE;
    x->x_last_score = 0.0f;
    x->x_has_score = 0;
    x->x_payload = NULL;
    x->x_payload_len = 0;
    x->x_payload_cap = 0;
    x->x_has_payload = 0;
    x->x_payload_sel = NULL;

    /* Optional creation args: <threshold> [<mode>] */
    int ai = 0;
    if (argc > ai && argv[ai].a_type == A_FLOAT) {
        x->x_threshold = (float)atom_getfloat(&argv[ai]);
        ai++;
    }
    if (argc > ai && argv[ai].a_type == A_SYMBOL) {
        const char *name = atom_getsymbol(&argv[ai])->s_name;
        if (strcmp(name, "above") == 0)      x->x_mode = NS_SPIGOT_ABOVE;
        else if (strcmp(name, "below") == 0) x->x_mode = NS_SPIGOT_BELOW;
    }

    /* Right inlet (proxy). */
    x->x_proxy.p_pd = ns_spigot_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlets: left = anything (preserves message selector), right = bang. */
    x->x_pass_out = outlet_new(&x->x_obj, &s_anything);
    x->x_block_out = outlet_new(&x->x_obj, &s_bang);

    return (void *)x;
}

static void ns_spigot_free(t_ns_spigot *x) {
    if (x->x_payload) {
        freebytes(x->x_payload, sizeof(t_atom) * x->x_payload_cap);
    }
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_spigot_setup(void) {
    ns_spigot_proxy_class = class_new(gensym("_ns_spigot_proxy"),
        0, 0, sizeof(t_ns_spigot_proxy), CLASS_PD, 0);
    class_addlist(ns_spigot_proxy_class, ns_spigot_proxy_list);
    class_addanything(ns_spigot_proxy_class, ns_spigot_proxy_anything);
    class_addmethod(ns_spigot_proxy_class, (t_method)ns_spigot_proxy_threshold,
                    gensym("threshold"), A_FLOAT, 0);
    class_addmethod(ns_spigot_proxy_class, (t_method)ns_spigot_proxy_mode,
                    gensym("mode"), A_SYMBOL, 0);
    class_addbang(ns_spigot_proxy_class, ns_spigot_proxy_bang);

    ns_spigot_class = class_new(gensym("ns_spigot"),
        (t_newmethod)ns_spigot_new,
        (t_method)ns_spigot_free,
        sizeof(t_ns_spigot),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addfloat(ns_spigot_class, ns_spigot_float);

    post("ns_spigot %s - threshold-gated list passer", NS_VERSION_STRING);
}
