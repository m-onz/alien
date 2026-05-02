/*
 * ns_quality - Domain-agnostic quality score for a rendered alien sequence
 *
 *   [ns_quality]
 *
 * Takes a sequence list (the same shape [alien] emits — floats for values,
 * the symbol `-` for rests) and returns a quality score in [0, 1].
 *
 * Designed to live between [alien] and [ns_corpus] as the "quality" half
 * of an admission pair. ns_archive owns novelty; ns_quality owns "is this
 * even worth keeping?" The variety component is intentionally domain-
 * agnostic — distinct numeric values, no pitch-class assumptions — so the
 * same external works whether the numbers in the sequence map to MIDI
 * notes, sample-folder indices, or arbitrary parameter values.
 *
 * Hot left inlet (list):
 *     A sequence as emitted by [alien]. Floats become note values; the
 *     symbol `-` (or any non-float atom) becomes a rest.
 *
 * Outlets (left → right, decreasing detail):
 *     left  (float):  combined quality score in [0, 1]
 *     right (list):   the four components, for logging/debug —
 *                     <rest_ratio> <length_score> <variety_score> <validity>
 *                     where validity is 1.0 if the hard gates passed,
 *                     0.0 if any failed (in which case score is also 0).
 *
 * Scoring (matches the design we agreed on):
 *
 *   Hard gates (score = 0 if any fails):
 *     - length < 4
 *     - all rests
 *     - any value is non-finite (NaN, ±inf — alien occasionally emits
 *       these from out-of-range drunk/scale params)
 *
 *   Soft components (each in [0, 1]):
 *     - rest_ratio_fitness  piecewise — 1.0 in [0.20, 0.50], 0.6 at 0%
 *                           rests (dense is fine, just monotone), tapering
 *                           down to 0.0 only as we approach all-rest
 *     - length_score        0.6 below 4, ramp to 1.0 at 8, hold to 64,
 *                           taper to 0.6 by 256
 *     - variety_score       distinct_values / num_non_rest_values,
 *                           capped at 1.0 (a single-note rhythm like
 *                           `(rep 42 16)` scores 1/16 here, which is
 *                           low but valid — variety is a *soft* term,
 *                           never a hard reject)
 *
 *   Combined: 0.4 * rest + 0.3 * length + 0.3 * variety
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>
#include <math.h>

/* ======================================================================== */

#define NS_Q_MIN_LEN 4
#define NS_Q_MAX_INSPECT 1024  /* don't iterate past this many atoms */

static t_class *ns_quality_class;

typedef struct _ns_quality {
    t_object x_obj;
    t_outlet *x_out_score;
    t_outlet *x_out_components;
} t_ns_quality;

/* ======================================================================== */
/* SCORING — pure functions, no Pd dependencies                              */
/* ======================================================================== */

/* Linear interpolation between (x0,y0) and (x1,y1), clamped to the y-band. */
static inline float lerp_clamped(float x, float x0, float y0, float x1, float y1) {
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return y0 + t * (y1 - y0);
}

/* rest_ratio_fitness — matches the Python prototype's output_score curve:
 *   0.00       → 0.6   (dense / all-notes is fine, just monotone)
 *   0.00–0.20  → ramp  0.6 → 1.0
 *   0.20–0.50  → 1.0   (sweet spot — typical musical density)
 *   0.50–0.90  → ramp  1.0 → 0.1
 *   0.90–1.00  → ramp  0.1 → 0.0
 *
 * The asymmetry (0.6 at no-rests vs 0.0 at all-rests) is deliberate:
 * a monotone hihat-style pattern is a legitimate seed, but a sequence
 * of nothing but rests has no musical content at all.
 */
static float rest_fitness(float rest_ratio) {
    if (rest_ratio <  0.20f) return lerp_clamped(rest_ratio, 0.00f, 0.6f, 0.20f, 1.0f);
    if (rest_ratio <= 0.50f) return 1.0f;
    if (rest_ratio <  0.90f) return lerp_clamped(rest_ratio, 0.50f, 1.0f, 0.90f, 0.1f);
    return                          lerp_clamped(rest_ratio, 0.90f, 0.1f, 1.00f, 0.0f);
}

/* length_score:
 *   < 4    → 0.6  (we hard-reject below 4 anyway, but keep the soft band
 *                  defined so callers using this without hard gates get a
 *                  monotonic curve)
 *   4–8    → ramp 0.6 → 1.0
 *   8–64   → 1.0
 *   64–256 → ramp 1.0 → 0.6
 *   > 256  → 0.6
 */
static float length_score(int n) {
    float fn = (float)n;
    if (fn < 4.0f)   return 0.6f;
    if (fn < 8.0f)   return lerp_clamped(fn, 4.0f, 0.6f, 8.0f, 1.0f);
    if (fn <= 64.0f) return 1.0f;
    if (fn < 256.0f) return lerp_clamped(fn, 64.0f, 1.0f, 256.0f, 0.6f);
    return 0.6f;
}

/* variety_score: distinct_values / num_non_rest_values, capped at 1.0.
 * O(n²) inner loop is fine — sequences cap at NS_Q_MAX_INSPECT. */
static float variety_score(const int *notes, int n_notes) {
    if (n_notes <= 0) return 0.0f;
    int distinct = 0;
    for (int i = 0; i < n_notes; i++) {
        int dup = 0;
        for (int j = 0; j < i; j++) {
            if (notes[j] == notes[i]) { dup = 1; break; }
        }
        if (!dup) distinct++;
    }
    float v = (float)distinct / (float)n_notes;
    if (v > 1.0f) v = 1.0f;
    return v;
}

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_quality_list(t_ns_quality *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;

    /* Truncate inspection to NS_Q_MAX_INSPECT. Any longer is fine for
     * scoring purposes — we don't want to allocate proportional buffers. */
    int n = (argc > NS_Q_MAX_INSPECT) ? NS_Q_MAX_INSPECT : argc;

    /* Pass 1: classify each atom, count rests, detect non-finite, collect
     * notes into a stack buffer for variety. */
    int notes[NS_Q_MAX_INSPECT];
    int n_notes = 0;
    int n_rest = 0;
    int has_bad = 0;

    for (int i = 0; i < n; i++) {
        if (argv[i].a_type == A_FLOAT) {
            float f = (float)atom_getfloat(&argv[i]);
            if (!isfinite(f)) {
                has_bad = 1;
                continue;
            }
            /* int-cast for variety comparison; rounded to nearest. */
            int v = (f >= 0.0f) ? (int)(f + 0.5f) : -(int)(-f + 0.5f);
            notes[n_notes++] = v;
        } else {
            /* Symbol or anything non-numeric → rest. */
            n_rest++;
        }
    }

    /* Hard gates. */
    int hard_pass = 1;
    if (n < NS_Q_MIN_LEN)        hard_pass = 0;  /* too short */
    if (n > 0 && n_rest == n)    hard_pass = 0;  /* all rests */
    if (has_bad)                 hard_pass = 0;  /* non-finite values */

    float rest_ratio = (n > 0) ? (float)n_rest / (float)n : 1.0f;
    float r_score = hard_pass ? rest_fitness(rest_ratio) : 0.0f;
    float l_score = hard_pass ? length_score(n)         : 0.0f;
    float v_score = hard_pass ? variety_score(notes, n_notes) : 0.0f;

    float final = hard_pass
        ? (0.4f * r_score + 0.3f * l_score + 0.3f * v_score)
        : 0.0f;

    /* Right outlet first so that — once the score lands on the left and
     * triggers downstream effects — the components have already been
     * stamped wherever they were going. Pd executes right-to-left on
     * outlets within a single message; this keeps logging deterministic. */
    t_atom comp[4];
    SETFLOAT(&comp[0], (t_float)rest_ratio);
    SETFLOAT(&comp[1], (t_float)l_score);
    SETFLOAT(&comp[2], (t_float)v_score);
    SETFLOAT(&comp[3], (t_float)(hard_pass ? 1.0f : 0.0f));
    outlet_list(x->x_out_components, &s_list, 4, comp);

    outlet_float(x->x_out_score, (t_float)final);
}

/* Bare floats: treat as a 1-element list (will fail the length gate). */
static void ns_quality_float(t_ns_quality *x, t_floatarg f) {
    t_atom a;
    SETFLOAT(&a, f);
    ns_quality_list(x, &s_list, 1, &a);
}

/* Empty bang: emit zeros so downstream stays in sync. */
static void ns_quality_bang(t_ns_quality *x) {
    ns_quality_list(x, &s_list, 0, NULL);
}

/* ======================================================================== */
/* CONSTRUCTOR / SETUP                                                      */
/* ======================================================================== */

static void *ns_quality_new(void) {
    t_ns_quality *x = (t_ns_quality *)pd_new(ns_quality_class);
    x->x_out_score      = outlet_new(&x->x_obj, &s_float);
    x->x_out_components = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}

void ns_quality_setup(void) {
    ns_quality_class = class_new(gensym("ns_quality"),
        (t_newmethod)ns_quality_new,
        0,
        sizeof(t_ns_quality),
        CLASS_DEFAULT,
        0);

    class_addlist(ns_quality_class, ns_quality_list);
    class_addfloat(ns_quality_class, ns_quality_float);
    class_addbang(ns_quality_class, ns_quality_bang);

    post("ns_quality %s - rendered sequence → quality score [0,1]",
         NS_VERSION_STRING);
}
