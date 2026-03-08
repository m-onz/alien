/*
 * alien_eg_router - Entangled Group router
 *
 * Like [alien_router_entangled] but with a namespace/group argument.
 * Only clears [alien_eg] instances with matching group.
 *
 * Usage: [alien_eg_router drums kick snare hihat]
 *        [alien_eg_router synths bass lead pad]
 *
 * First argument is the group name, remaining are routing symbols.
 *
 * Inlets:
 *   1 (left): messages to route (e.g., "kick (euclid 4 16)")
 *   2 (right): bang - clear matching group + flush buffered patterns
 *
 * Flush sequence:
 *   1. Broadcast "clear" to all [alien_eg <group>] instances
 *   2. Output all buffered patterns simultaneously
 */

#include "m_pd.h"
#include "alien_core.h"
#include <string.h>

static t_class *alien_eg_router_class;

#define MAX_PATTERN_LEN 256

typedef struct _router_buffer {
    t_atom data[MAX_PATTERN_LEN];
    int len;
    int has_data;
} t_router_buffer;

typedef struct _alien_eg_router {
    t_object x_obj;
    t_inlet *x_flush_inlet;
    t_symbol *x_group;          // group name
    t_symbol *x_bind_sym;       // computed bind symbol "__alien_eg_<group>__"
    int x_n;                    // number of routing symbols
    t_symbol **x_vec;           // array of symbols to match
    t_outlet **x_outlets;       // array of outlets (n+1)
    t_router_buffer *x_buffers; // array of buffers (n+1)
} t_alien_eg_router;

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

static void alien_eg_router_anything(t_alien_eg_router *x, t_symbol *s, int argc, t_atom *argv) {
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
// FLUSH - Clear group, then output buffered patterns
// ============================================================================

static void alien_eg_router_flush(t_alien_eg_router *x) {
    // Step 1: Broadcast "clear" to all alien_eg instances in this group
    if (x->x_bind_sym->s_thing) {
        pd_typedmess(x->x_bind_sym->s_thing, gensym("clear"), 0, NULL);
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

static void alien_eg_router_clear(t_alien_eg_router *x) {
    if (x->x_bind_sym->s_thing) {
        pd_typedmess(x->x_bind_sym->s_thing, gensym("clear"), 0, NULL);
    }
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_eg_router_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien_eg_router *x = (t_alien_eg_router *)pd_new(alien_eg_router_class);

    // First argument is group name
    if (argc > 0 && argv[0].a_type == A_SYMBOL) {
        x->x_group = atom_getsymbol(&argv[0]);
        argv++;
        argc--;
    } else {
        x->x_group = gensym("default");
    }
    
    // Create bind symbol: "__alien_eg_<group>__"
    char bind_name[256];
    snprintf(bind_name, sizeof(bind_name), "__alien_eg_%s__", x->x_group->s_name);
    x->x_bind_sym = gensym(bind_name);

    // Remaining arguments are routing symbols
    x->x_n = argc;

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

    // Store the routing symbols
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            x->x_vec[i] = atom_getsymbol(&argv[i]);
        } else {
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

    // Create outlets
    for (int i = 0; i <= argc; i++) {
        x->x_outlets[i] = outlet_new(&x->x_obj, &s_list);
    }

    return (void *)x;
}

static void alien_eg_router_free(t_alien_eg_router *x) {
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

void alien_eg_router_setup(void) {
    alien_eg_router_class = class_new(gensym("alien_eg_router"),
        (t_newmethod)alien_eg_router_new,
        (t_method)alien_eg_router_free,
        sizeof(t_alien_eg_router),
        CLASS_DEFAULT,
        A_GIMME,
        0);
    class_addanything(alien_eg_router_class, alien_eg_router_anything);
    class_addmethod(alien_eg_router_class, (t_method)alien_eg_router_flush,
                    gensym("flush"), 0);
    class_addmethod(alien_eg_router_class, (t_method)alien_eg_router_clear,
                    gensym("clear"), 0);
    post("alien_eg_router %s - entangled group router", ALIEN_VERSION_STRING);
}
