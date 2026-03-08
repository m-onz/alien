/*
 * alien_router2 - Route messages by first symbol, output entire pattern
 *
 * Like Pd's [route] but passes entire remaining message instead of just first chunk.
 * Useful for routing pattern expressions like "kick (seq 1 2 3)" where you want
 * the full "(seq 1 2 3)" to be output, not just "(seq".
 *
 * Inlets:
 *   1 (left): messages to route
 *   2 (right): bang - flush all buffered patterns simultaneously
 *
 * Usage: [alien_router2 kick drum snare]
 * Creates N+1 outlets: one for each specified symbol, plus one for unmatched
 *
 * Buffered mode:
 *   Messages are stored until a bang on the right inlet triggers all outputs
 *   at once. This ensures all patterns are sent simultaneously for sync.
 *
 * Example:
 *   [alien_router2 kick snare]
 *   receives: kick (seq 1 2 3)   <- stored
 *   receives: snare (euclid 3 8) <- stored
 *   receives: bang on right inlet
 *   outputs both patterns simultaneously
 */

#include "m_pd.h"
#include "alien_core.h"
#include <string.h>

static t_class *alien_router2_class;

#define MAX_PATTERN_LEN 256

typedef struct _router_buffer {
    t_atom data[MAX_PATTERN_LEN];
    int len;
    int has_data;
} t_router_buffer;

typedef struct _alien_router2 {
    t_object x_obj;
    t_inlet *x_flush_inlet;
    int x_n;                    // number of routing symbols
    t_symbol **x_vec;           // array of symbols to match
    t_outlet **x_outlets;       // array of outlets (n+1)
    t_router_buffer *x_buffers; // array of buffers (n+1)
} t_alien_router2;

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

static void alien_router2_anything(t_alien_router2 *x, t_symbol *s, int argc, t_atom *argv) {
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
// FLUSH - Output all buffered patterns simultaneously
// ============================================================================

static void alien_router2_flush(t_alien_router2 *x) {
    // Output all buffered patterns (in reverse order for Pd right-to-left convention)
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
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_router2_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused parameter (required by Pd API)
    t_alien_router2 *x = (t_alien_router2 *)pd_new(alien_router2_class);

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

static void alien_router2_free(t_alien_router2 *x) {
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

void alien_router2_setup(void) {
    alien_router2_class = class_new(gensym("alien_router2"),
        (t_newmethod)alien_router2_new,
        (t_method)alien_router2_free,
        sizeof(t_alien_router2),
        CLASS_DEFAULT,
        A_GIMME,
        0);
    class_addanything(alien_router2_class, alien_router2_anything);
    class_addmethod(alien_router2_class, (t_method)alien_router2_flush,
                    gensym("flush"), 0);
    post("alien_router2 %s - pattern message router (buffered)", ALIEN_VERSION_STRING);
}
