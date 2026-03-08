/*
 * alien_router_entangled - Route messages with global clear broadcast
 *
 * Like [alien_router2] but when flushing, first broadcasts a "clear" message
 * to all [alien_entangled] instances, then outputs the buffered patterns.
 *
 * This ensures all patterns are cleared before new ones are set,
 * enabling clean section transitions without manual clearing.
 *
 * Inlets:
 *   1 (left): messages to route (e.g., "kick (euclid 4 16)")
 *   2 (right): bang - clear all entangled + flush buffered patterns
 *
 * Usage: [alien_router_entangled kick snare bass]
 * Creates N+1 outlets: one for each specified symbol, plus one for unmatched
 *
 * Flush sequence:
 *   1. Broadcast "clear" to all [alien_entangled] instances
 *   2. Output all buffered patterns simultaneously
 */

#include "m_pd.h"
#include "alien_core.h"
#include <string.h>

static t_class *alien_router_entangled_class;

// Global symbol for entanglement communication (shared with alien_entangled)
static t_symbol *s_entangle = NULL;

#define MAX_PATTERN_LEN 256

typedef struct _router_buffer {
    t_atom data[MAX_PATTERN_LEN];
    int len;
    int has_data;
} t_router_buffer;

typedef struct _alien_router_entangled {
    t_object x_obj;
    t_inlet *x_flush_inlet;
    int x_n;                    // number of routing symbols
    t_symbol **x_vec;           // array of symbols to match
    t_outlet **x_outlets;       // array of outlets (n+1)
    t_router_buffer *x_buffers; // array of buffers (n+1)
} t_alien_router_entangled;

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

static void alien_router_entangled_anything(t_alien_router_entangled *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the selector matches any of our routing symbols
    for (int i = 0; i < x->x_n; i++) {
        if (s == x->x_vec[i]) {
            // Match found - buffer the pattern
            t_router_buffer *buf = &x->x_buffers[i];
            int len = argc;
            if (len > MAX_PATTERN_LEN) len = MAX_PATTERN_LEN;
            
            for (int j = 0; j < len; j++) {
                buf->data[j] = argv[j];
            }
            buf->len = len;
            buf->has_data = 1;
            return;
        }
    }

    // No match found - buffer to the last slot (unmatched)
    // Include the selector as first element
    t_router_buffer *buf = &x->x_buffers[x->x_n];
    int len = argc + 1;
    if (len > MAX_PATTERN_LEN) len = MAX_PATTERN_LEN;
    
    SETSYMBOL(&buf->data[0], s);
    for (int j = 0; j < len - 1 && j < argc; j++) {
        buf->data[j + 1] = argv[j];
    }
    buf->len = len;
    buf->has_data = 1;
}

// ============================================================================
// FLUSH - Clear all entangled, then output buffered patterns
// ============================================================================

static void alien_router_entangled_flush(t_alien_router_entangled *x) {
    // Step 1: Broadcast "clear" to all alien_entangled instances
    if (s_entangle->s_thing) {
        pd_typedmess(s_entangle->s_thing, gensym("clear"), 0, NULL);
    }
    
    // Step 2: Output all buffered patterns (in reverse order for Pd right-to-left convention)
    for (int i = x->x_n; i >= 0; i--) {
        t_router_buffer *buf = &x->x_buffers[i];
        if (buf->has_data && buf->len > 0) {
            if (buf->data[0].a_type == A_SYMBOL) {
                outlet_anything(x->x_outlets[i], atom_getsymbol(&buf->data[0]), 
                               buf->len - 1, &buf->data[1]);
            } else {
                outlet_list(x->x_outlets[i], &s_list, buf->len, buf->data);
            }
        }
    }
}

// ============================================================================
// CLEAR ONLY - Just broadcast clear without flushing
// ============================================================================

static void alien_router_entangled_clear(t_alien_router_entangled *x) {
    (void)x;  // unused
    if (s_entangle->s_thing) {
        pd_typedmess(s_entangle->s_thing, gensym("clear"), 0, NULL);
    }
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_router_entangled_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused parameter
    t_alien_router_entangled *x = (t_alien_router_entangled *)pd_new(alien_router_entangled_class);

    // Store the number of routing symbols
    x->x_n = argc;

    // Allocate arrays for symbols, outlets, and buffers
    if (argc > 0) {
        x->x_vec = (t_symbol **)getbytes(argc * sizeof(t_symbol *));
    } else {
        x->x_vec = NULL;
    }
    x->x_outlets = (t_outlet **)getbytes((argc + 1) * sizeof(t_outlet *));
    x->x_buffers = (t_router_buffer *)getbytes((argc + 1) * sizeof(t_router_buffer));
    
    // Initialize buffers
    for (int i = 0; i <= argc; i++) {
        x->x_buffers[i].len = 0;
        x->x_buffers[i].has_data = 0;
    }

    // Store the routing symbols from creation arguments
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            x->x_vec[i] = atom_getsymbol(&argv[i]);
        } else {
            // If argument is not a symbol, convert it to symbol
            char buf[32];
            if (argv[i].a_type == A_FLOAT) {
                float f = atom_getfloat(&argv[i]);
                if (f == (int)f) {
                    snprintf(buf, sizeof(buf), "%d", (int)f);
                } else {
                    snprintf(buf, sizeof(buf), "%g", f);
                }
            } else {
                snprintf(buf, sizeof(buf), "?");
            }
            x->x_vec[i] = gensym(buf);
        }
    }

    // Create flush inlet (right inlet)
    x->x_flush_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_bang, gensym("flush"));

    // Create outlets: one for each routing symbol, plus one for unmatched
    for (int i = 0; i <= argc; i++) {
        x->x_outlets[i] = outlet_new(&x->x_obj, &s_list);
    }

    return (void *)x;
}

static void alien_router_entangled_free(t_alien_router_entangled *x) {
    if (x->x_n > 0) {
        freebytes(x->x_vec, x->x_n * sizeof(t_symbol *));
    }
    freebytes(x->x_outlets, (x->x_n + 1) * sizeof(t_outlet *));
    freebytes(x->x_buffers, (x->x_n + 1) * sizeof(t_router_buffer));
    inlet_free(x->x_flush_inlet);
}

// ============================================================================
// SETUP
// ============================================================================

void alien_router_entangled_setup(void) {
    // Initialize global entanglement symbol (shared with alien_entangled)
    s_entangle = gensym("__alien_entangle__");
    
    alien_router_entangled_class = class_new(gensym("alien_router_entangled"),
        (t_newmethod)alien_router_entangled_new,
        (t_method)alien_router_entangled_free,
        sizeof(t_alien_router_entangled),
        CLASS_DEFAULT,
        A_GIMME,
        0);
    class_addanything(alien_router_entangled_class, alien_router_entangled_anything);
    class_addmethod(alien_router_entangled_class, (t_method)alien_router_entangled_flush,
                    gensym("flush"), 0);
    class_addmethod(alien_router_entangled_class, (t_method)alien_router_entangled_clear,
                    gensym("clear"), 0);
    post("alien_router_entangled %s - pattern router with global sync", ALIEN_VERSION_STRING);
}
