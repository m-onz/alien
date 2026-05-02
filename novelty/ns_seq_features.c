/*
 * ns_seq_features - Sequence (notes + rests) → 27-dim BC for Pure Data
 *
 *   [ns_seq_features]
 *
 * Designed for the output of [alien]: a list whose elements are either
 * floats (note values, typically MIDI 0–127) or the symbol `-` for a rest.
 * Outputs a 27-dim feature vector — a port of novelty_search.py's embed().
 *
 * Hot left inlet (list):
 *     A sequence. Floats become note values (rounded to int). The symbol
 *     `-` becomes a rest. Any other atom type is treated as a rest.
 *
 * Outlet:
 *     left:  list of 27 floats (the BC). Layout, in order:
 *
 *         0   length_norm        len / 96, capped
 *         1   rest_ratio         fraction of rests
 *         2   unique_ratio       |unique notes| / |notes|
 *         3   mean_norm          mean / 127
 *         4   pitch_range        (max - min) / 127
 *         5   pitch_std          std / 32
 *         6   ascend             fraction of intervals > 0
 *         7   descend            fraction of intervals < 0
 *         8   repeat             fraction of intervals = 0
 *         9   rhythm_ent         note/rest run-length Shannon entropy
 *         10  ac1                lag-1 mask autocorr → [0, 1]
 *         11  ac2                lag-2 mask autocorr → [0, 1]
 *         12-26  hist[15]        interval histogram (bins:
 *                                -24, -12, -7, -5, -3, -2, -1,
 *                                 0, 1, 2, 3, 5, 7, 12, 24)
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>

/* ======================================================================== */

static t_class *ns_seq_features_class;

typedef struct _ns_seq_features {
    t_object x_obj;
    t_outlet *x_out;
} t_ns_seq_features;

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_seq_features_list(t_ns_seq_features *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;

    if (argc <= 0) {
        /* Empty input → emit zero vector so the patch sees something. */
        t_atom out_atoms[NS_SEQ_FEATURE_DIM];
        for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) SETFLOAT(&out_atoms[i], 0.0f);
        outlet_list(x->x_out, &s_list, NS_SEQ_FEATURE_DIM, out_atoms);
        return;
    }

    int n = (argc > NS_SEQ_MAX_LEN) ? NS_SEQ_MAX_LEN : argc;

    int seq[NS_SEQ_MAX_LEN];
    for (int i = 0; i < n; i++) {
        if (argv[i].a_type == A_FLOAT) {
            float f = (float)atom_getfloat(&argv[i]);
            /* Defensive: clamp to plausible musical range to keep BC math
             * bounded. Upstream evaluators (e.g. alien drunk with a hyphen
             * start arg) can return INT_MAX / INT_MIN garbage that would
             * otherwise blow up pitch_std and explode kNN distances. */
            if (f < -127.0f || f > 127.0f) {
                seq[i] = NS_REST;
            } else {
                seq[i] = (f >= 0.0f) ? (int)(f + 0.5f) : -(int)(-f + 0.5f);
            }
        } else {
            /* Symbol or anything else → rest. */
            seq[i] = NS_REST;
        }
    }

    float feat[NS_SEQ_FEATURE_DIM];
    ns_seq_features(seq, n, feat);

    t_atom out_atoms[NS_SEQ_FEATURE_DIM];
    for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) {
        SETFLOAT(&out_atoms[i], (t_float)feat[i]);
    }
    outlet_list(x->x_out, &s_list, NS_SEQ_FEATURE_DIM, out_atoms);
}

/* Bare floats: treat as a 1-element list. */
static void ns_seq_features_float(t_ns_seq_features *x, t_floatarg f) {
    t_atom a;
    SETFLOAT(&a, f);
    ns_seq_features_list(x, &s_list, 1, &a);
}

/* ======================================================================== */
/* CONSTRUCTOR                                                              */
/* ======================================================================== */

static void *ns_seq_features_new(void) {
    t_ns_seq_features *x = (t_ns_seq_features *)pd_new(ns_seq_features_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_seq_features_setup(void) {
    ns_seq_features_class = class_new(gensym("ns_seq_features"),
        (t_newmethod)ns_seq_features_new,
        0,
        sizeof(t_ns_seq_features),
        CLASS_DEFAULT,
        0);

    class_addlist(ns_seq_features_class, ns_seq_features_list);
    class_addfloat(ns_seq_features_class, ns_seq_features_float);

    post("ns_seq_features %s - sequence (notes + rests) → 27-dim BC", NS_VERSION_STRING);
}
