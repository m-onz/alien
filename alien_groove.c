/*
 * alien_groove.c - Stochastic pattern filter for Pure Data
 *
 * Applies a stochastic mask to incoming patterns.
 *
 * Inlets:
 *   1 (left, hot): list - pattern to filter
 *   2 (right, cold): list - template pattern (optional)
 *
 * Outlets:
 *   1: list - filtered pattern
 *
 * Messages:
 *   strictness <0-100>   - probability of dropping notes (default 50)
 *   phase <n>            - rotate template by N steps (default 0)
 *
 * Behavior:
 *   With template: notes on template positions pass through,
 *                  notes off template are dropped with probability = strictness%
 *   Without template: all notes are dropped with probability = strictness%
 */

#include "m_pd.h"
#include <stdlib.h>
#include <time.h>

#define ALIEN_VERSION_STRING "0.2.1"

static t_class *alien_groove_class;

#define MAX_PATTERN_LEN 256

typedef struct _alien_groove {
    t_object x_obj;
    t_inlet *x_template_inlet;
    t_outlet *x_out;

    int x_template[MAX_PATTERN_LEN];   // Template pattern (1=hit, 0=rest)
    int x_template_len;

    int x_strictness;                   // 0-100 (probability of dropping)
    int x_phase;                        // Template rotation offset

    int x_random_initialized;
} t_alien_groove;

// ============================================================================
// RANDOM UTILITIES
// ============================================================================

static void ensure_random_init(t_alien_groove *x) {
    if (!x->x_random_initialized) {
        srand((unsigned int)time(NULL));
        x->x_random_initialized = 1;
    }
}

static int random_percent(t_alien_groove *x) {
    ensure_random_init(x);
    return rand() % 100;
}

// ============================================================================
// TEMPLATE HELPERS
// ============================================================================

static int template_is_hit(t_alien_groove *x, int i) {
    if (x->x_template_len == 0) return 0;  // No template = apply stochastic filter to all
    int idx = ((i + x->x_phase) % x->x_template_len + x->x_template_len)
              % x->x_template_len;
    return x->x_template[idx] != 0;
}

// ============================================================================
// MESSAGE HANDLERS
// ============================================================================

static void alien_groove_list(t_alien_groove *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc == 0) return;
    if (argc > MAX_PATTERN_LEN) argc = MAX_PATTERN_LEN;

    t_atom *out = (t_atom *)getbytes(sizeof(t_atom) * argc);

    for (int i = 0; i < argc; i++) {
        int is_rest = (argv[i].a_type == A_SYMBOL) ||
                      (argv[i].a_type == A_FLOAT && atom_getfloat(&argv[i]) == -1);

        if (is_rest) {
            // Rests pass through unchanged
            out[i] = argv[i];
        } else if (x->x_template_len > 0 && template_is_hit(x, i)) {
            // With template: notes on template positions always pass
            out[i] = argv[i];
        } else {
            // No template OR note off template: apply stochastic filter
            if (random_percent(x) < x->x_strictness) {
                SETSYMBOL(&out[i], gensym("-"));
            } else {
                out[i] = argv[i];
            }
        }
    }

    outlet_list(x->x_out, &s_list, argc, out);
    freebytes(out, sizeof(t_atom) * argc);
}

static void alien_groove_template(t_alien_groove *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    x->x_template_len = (argc > MAX_PATTERN_LEN) ? MAX_PATTERN_LEN : argc;

    for (int i = 0; i < x->x_template_len; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            x->x_template[i] = 0;
        } else {
            float val = atom_getfloat(&argv[i]);
            x->x_template[i] = (val != 0 && val != -1) ? 1 : 0;
        }
    }
}

static void alien_groove_strictness(t_alien_groove *x, t_floatarg f) {
    x->x_strictness = (int)f;
    if (x->x_strictness < 0) x->x_strictness = 0;
    if (x->x_strictness > 100) x->x_strictness = 100;
}

static void alien_groove_phase(t_alien_groove *x, t_floatarg f) {
    x->x_phase = (int)f;
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_groove_new(void) {
    t_alien_groove *x = (t_alien_groove *)pd_new(alien_groove_class);

    x->x_strictness = 50;
    x->x_phase = 0;
    x->x_template_len = 0;
    x->x_random_initialized = 0;

    x->x_template_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_list, gensym("template"));
    x->x_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

static void alien_groove_free(t_alien_groove *x) {
    inlet_free(x->x_template_inlet);
}

// ============================================================================
// SETUP
// ============================================================================

void alien_groove_setup(void) {
    alien_groove_class = class_new(
        gensym("alien_groove"),
        (t_newmethod)alien_groove_new,
        (t_method)alien_groove_free,
        sizeof(t_alien_groove),
        CLASS_DEFAULT,
        0
    );

    class_addlist(alien_groove_class, alien_groove_list);
    class_addmethod(alien_groove_class, (t_method)alien_groove_template,
                    gensym("template"), A_GIMME, 0);
    class_addmethod(alien_groove_class, (t_method)alien_groove_strictness,
                    gensym("strictness"), A_FLOAT, 0);
    class_addmethod(alien_groove_class, (t_method)alien_groove_phase,
                    gensym("phase"), A_FLOAT, 0);
    post("alien_groove %s - stochastic pattern filter", ALIEN_VERSION_STRING);
}
