/*
 * alien_groove.c - Rhythmic pattern constrainer for Pure Data
 *
 * Constrains incoming rhythmic patterns to a template groove
 * with variable strictness.
 *
 * Inlets:
 *   1 (left, hot): list - pattern to constrain
 *   2 (right, cold): list - template pattern
 *
 * Outlets:
 *   1: list - constrained pattern
 *
 * Messages:
 *   strictness <0-100>   - how strictly to enforce (default 100)
 *   mode <mask|pull|push> - constraint mode (default mask)
 *
 * Modes:
 *   mask - silence non-aligned hits based on strictness
 *   pull - pull hits toward nearest template beat
 *   push - push hits away from template (counter-rhythm)
 */

#include "m_pd.h"
#include "alien_core.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static t_class *alien_groove_class;

#define MAX_PATTERN_LEN 256
#define REST_VALUE -1

typedef enum {
    MODE_MASK,   // Silence non-aligned hits based on strictness
    MODE_PULL,   // Pull hits toward template positions
    MODE_PUSH    // Push hits away from template (inverse/counter-rhythm)
} groove_mode;

typedef struct _alien_groove {
    t_object x_obj;
    t_inlet *x_template_inlet;
    t_outlet *x_out;

    int x_template[MAX_PATTERN_LEN];   // Template pattern (1=hit, 0=rest)
    int x_template_len;

    int x_strictness;                   // 0-100
    groove_mode x_mode;

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

// Check if position i in template is a hit (cycling if needed)
static int template_is_hit(t_alien_groove *x, int i) {
    if (x->x_template_len == 0) return 1;  // No template = everything allowed
    int idx = i % x->x_template_len;
    return x->x_template[idx] != 0;
}

// Find nearest template hit position (for pull mode)
static int nearest_template_hit(t_alien_groove *x, int pos, int pattern_len) {
    if (x->x_template_len == 0) return pos;

    // Search outward from pos
    for (int offset = 0; offset <= pattern_len; offset++) {
        int left = pos - offset;
        int right = pos + offset;

        if (left >= 0 && template_is_hit(x, left)) return left;
        if (right < pattern_len && template_is_hit(x, right)) return right;
    }
    return pos;  // Fallback
}

// Find nearest template rest position (for push mode)
static int nearest_template_rest(t_alien_groove *x, int pos, int pattern_len) {
    if (x->x_template_len == 0) return pos;

    for (int offset = 0; offset <= pattern_len; offset++) {
        int left = pos - offset;
        int right = pos + offset;

        if (left >= 0 && !template_is_hit(x, left)) return left;
        if (right < pattern_len && !template_is_hit(x, right)) return right;
    }
    return pos;
}

// ============================================================================
// CONSTRAINT ALGORITHMS
// ============================================================================

static void process_mask_mode(t_alien_groove *x, t_atom *in, t_atom *out, int len) {
    for (int i = 0; i < len; i++) {
        int is_rest = (in[i].a_type == A_SYMBOL) ||
                      (in[i].a_type == A_FLOAT && atom_getfloat(&in[i]) == REST_VALUE);

        if (is_rest) {
            // Rests pass through unchanged
            out[i] = in[i];
        } else if (template_is_hit(x, i)) {
            // Hit on template position: always passes
            out[i] = in[i];
        } else {
            // Hit on non-template position: probabilistic based on strictness
            if (random_percent(x) < x->x_strictness) {
                // Silence it - output "-" symbol, not -1
                SETSYMBOL(&out[i], gensym("-"));
            } else {
                // Let it through
                out[i] = in[i];
            }
        }
    }
}

static void process_pull_mode(t_alien_groove *x, t_atom *in, t_atom *out, int len) {
    // Initialize output as all rests (use "-" symbol, not -1)
    for (int i = 0; i < len; i++) {
        SETSYMBOL(&out[i], gensym("-"));
    }

    for (int i = 0; i < len; i++) {
        int is_rest = (in[i].a_type == A_SYMBOL) ||
                      (in[i].a_type == A_FLOAT && atom_getfloat(&in[i]) == REST_VALUE);

        if (is_rest) continue;

        if (template_is_hit(x, i)) {
            // Already on template: stays put
            out[i] = in[i];
        } else {
            // Not on template: maybe pull toward nearest template hit
            if (random_percent(x) < x->x_strictness) {
                int target = nearest_template_hit(x, i, len);
                // Only move if target is currently a rest (avoid collisions)
                if (atom_getfloat(&out[target]) == REST_VALUE) {
                    out[target] = in[i];
                }
                // Original position becomes rest (already is)
            } else {
                // Let it stay where it is
                out[i] = in[i];
            }
        }
    }
}

static void process_push_mode(t_alien_groove *x, t_atom *in, t_atom *out, int len) {
    // Initialize output as all rests (use "-" symbol, not -1)
    for (int i = 0; i < len; i++) {
        SETSYMBOL(&out[i], gensym("-"));
    }

    for (int i = 0; i < len; i++) {
        int is_rest = (in[i].a_type == A_SYMBOL) ||
                      (in[i].a_type == A_FLOAT && atom_getfloat(&in[i]) == REST_VALUE);

        if (is_rest) continue;

        if (!template_is_hit(x, i)) {
            // Already off template: stays put
            out[i] = in[i];
        } else {
            // On template: maybe push to nearest rest position
            if (random_percent(x) < x->x_strictness) {
                int target = nearest_template_rest(x, i, len);
                if (target != i && atom_getfloat(&out[target]) == REST_VALUE) {
                    out[target] = in[i];
                }
                // If can't push, it gets silenced (counter-rhythm effect)
            } else {
                out[i] = in[i];
            }
        }
    }
}

// ============================================================================
// MESSAGE HANDLERS
// ============================================================================

// Left inlet: pattern to constrain (hot)
static void alien_groove_list(t_alien_groove *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused
    if (argc == 0) return;
    if (argc > MAX_PATTERN_LEN) argc = MAX_PATTERN_LEN;

    t_atom *out = (t_atom *)getbytes(sizeof(t_atom) * argc);

    switch (x->x_mode) {
        case MODE_PULL:
            process_pull_mode(x, argv, out, argc);
            break;
        case MODE_PUSH:
            process_push_mode(x, argv, out, argc);
            break;
        case MODE_MASK:
        default:
            process_mask_mode(x, argv, out, argc);
            break;
    }

    outlet_list(x->x_out, &s_list, argc, out);
    freebytes(out, sizeof(t_atom) * argc);
}

// Right inlet: template pattern (cold)
static void alien_groove_template(t_alien_groove *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused
    x->x_template_len = (argc > MAX_PATTERN_LEN) ? MAX_PATTERN_LEN : argc;

    for (int i = 0; i < x->x_template_len; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            // Symbol (like "-") = rest = 0
            x->x_template[i] = 0;
        } else {
            // Any non-zero number (except REST_VALUE) = hit
            float val = atom_getfloat(&argv[i]);
            x->x_template[i] = (val != 0 && val != REST_VALUE) ? 1 : 0;
        }
    }
}

static void alien_groove_strictness(t_alien_groove *x, t_floatarg f) {
    x->x_strictness = (int)f;
    if (x->x_strictness < 0) x->x_strictness = 0;
    if (x->x_strictness > 100) x->x_strictness = 100;
}

static void alien_groove_mode(t_alien_groove *x, t_symbol *s) {
    if (strcmp(s->s_name, "mask") == 0) {
        x->x_mode = MODE_MASK;
    } else if (strcmp(s->s_name, "pull") == 0) {
        x->x_mode = MODE_PULL;
    } else if (strcmp(s->s_name, "push") == 0) {
        x->x_mode = MODE_PUSH;
    } else {
        pd_error(x, "alien_groove: unknown mode '%s' (use mask, pull, or push)", s->s_name);
    }
}

// ============================================================================
// CONSTRUCTOR
// ============================================================================

static void *alien_groove_new(void) {
    t_alien_groove *x = (t_alien_groove *)pd_new(alien_groove_class);

    // Defaults
    x->x_strictness = 100;
    x->x_mode = MODE_MASK;
    x->x_template_len = 0;  // No template = everything passes
    x->x_random_initialized = 0;

    // Create right inlet for template (cold)
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
    class_addmethod(alien_groove_class, (t_method)alien_groove_mode,
                    gensym("mode"), A_SYMBOL, 0);
    post("alien_groove %s - rhythmic pattern constrainer", ALIEN_VERSION_STRING);
}
