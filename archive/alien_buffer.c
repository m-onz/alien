/*
 * alien_buffer.c - Pattern buffer for Pure Data
 *
 * Stores a pattern and outputs it on demand.
 *
 * Inlets:
 *   1 (left): list - pattern to store
 *   2 (right): bang - trigger output
 *
 * Outlets:
 *   1: list - stored pattern
 *
 * Creation: [alien_buffer]
 */

#include "m_pd.h"

#define ALIEN_VERSION_STRING "0.2.1"
#define MAX_PATTERN_LEN 256

static t_class *alien_buffer_class;

typedef struct _alien_buffer {
    t_object x_obj;
    t_inlet *x_trigger_inlet;
    t_outlet *x_out;

    t_atom x_pattern[MAX_PATTERN_LEN];
    int x_len;
} t_alien_buffer;

// ============================================================================
// TRIGGER - Output stored pattern
// ============================================================================

static void alien_buffer_trigger(t_alien_buffer *x) {
    if (x->x_len > 0) {
        outlet_list(x->x_out, &s_list, x->x_len, x->x_pattern);
    }
}

// ============================================================================
// PATTERN INPUT
// ============================================================================

static void alien_buffer_list(t_alien_buffer *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc == 0) return;
    if (argc > MAX_PATTERN_LEN) argc = MAX_PATTERN_LEN;

    x->x_len = argc;
    for (int i = 0; i < argc; i++) {
        x->x_pattern[i] = argv[i];
    }
}

static void alien_buffer_float(t_alien_buffer *x, t_floatarg f) {
    t_atom a;
    SETFLOAT(&a, f);
    x->x_len = 1;
    x->x_pattern[0] = a;
}

static void alien_buffer_anything(t_alien_buffer *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc == 0) {
        t_atom a;
        SETSYMBOL(&a, s);
        x->x_len = 1;
        x->x_pattern[0] = a;
    } else {
        if (argc + 1 > MAX_PATTERN_LEN) argc = MAX_PATTERN_LEN - 1;
        SETSYMBOL(&x->x_pattern[0], s);
        for (int i = 0; i < argc; i++) {
            x->x_pattern[i + 1] = argv[i];
        }
        x->x_len = argc + 1;
    }
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_buffer_new(void) {
    t_alien_buffer *x = (t_alien_buffer *)pd_new(alien_buffer_class);

    x->x_len = 0;

    // Right inlet for trigger
    x->x_trigger_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_bang, gensym("trigger"));

    // Outlet
    x->x_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

static void alien_buffer_free(t_alien_buffer *x) {
    inlet_free(x->x_trigger_inlet);
}

// ============================================================================
// SETUP
// ============================================================================

void alien_buffer_setup(void) {
    alien_buffer_class = class_new(
        gensym("alien_buffer"),
        (t_newmethod)alien_buffer_new,
        (t_method)alien_buffer_free,
        sizeof(t_alien_buffer),
        CLASS_DEFAULT,
        0
    );

    class_addlist(alien_buffer_class, alien_buffer_list);
    class_addfloat(alien_buffer_class, alien_buffer_float);
    class_addanything(alien_buffer_class, alien_buffer_anything);
    class_addmethod(alien_buffer_class, (t_method)alien_buffer_trigger,
                    gensym("trigger"), 0);

    post("alien_buffer %s - pattern buffer", ALIEN_VERSION_STRING);
}
