/*
 * alien_router - Route messages by first symbol, output entire pattern
 *
 * Like Pd's [route] but passes entire remaining message instead of just first chunk.
 * Useful for routing pattern expressions like "kick (seq 1 2 3)" where you want
 * the full "(seq 1 2 3)" to be output, not just "(seq".
 *
 * Usage: [alien_router kick drum snare]
 * Creates N+1 outlets: one for each specified symbol, plus one for unmatched
 *
 * Example:
 *   [alien_router kick snare]
 *   receives: kick (seq 1 2 3)
 *   outputs "(seq 1 2 3)" from first outlet
 *   receives: snare (euclid 3 8)
 *   outputs "(euclid 3 8)" from second outlet
 *   receives: hat (seq 1 - 1 -)
 *   outputs "hat (seq 1 - 1 -)" from third outlet (unmatched)
 */

#include "m_pd.h"
#include <string.h>

static t_class *alien_router_class;

typedef struct _alien_router {
    t_object x_obj;
    int x_n;                    // number of routing symbols
    t_symbol **x_vec;           // array of symbols to match
    t_outlet **x_outlets;       // array of outlets (n+1)
} t_alien_router;

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

static void alien_router_anything(t_alien_router *x, t_symbol *s, int argc, t_atom *argv) {
    // Check if the selector matches any of our routing symbols
    for (int i = 0; i < x->x_n; i++) {
        if (s == x->x_vec[i]) {
            // Match found - output the remaining arguments to this outlet
            if (argc == 0) {
                // Just the symbol with no arguments - send bang
                outlet_bang(x->x_outlets[i]);
            } else if (argc == 1 && argv[0].a_type == A_FLOAT) {
                // Single float - send as float
                outlet_float(x->x_outlets[i], atom_getfloat(argv));
            } else if (argc == 1 && argv[0].a_type == A_SYMBOL) {
                // Single symbol - send as symbol (or anything message with no args)
                t_symbol *sym = atom_getsymbol(argv);
                outlet_anything(x->x_outlets[i], sym, 0, NULL);
            } else {
                // Multiple arguments - send as anything message
                // First argument becomes the selector, rest become arguments
                if (argv[0].a_type == A_SYMBOL) {
                    outlet_anything(x->x_outlets[i], atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
                } else {
                    // First arg is a float, send as list
                    outlet_list(x->x_outlets[i], &s_list, argc, argv);
                }
            }
            return;
        }
    }

    // No match found - output everything to the last outlet (unmatched)
    // Reconstruct as anything message with original selector
    outlet_anything(x->x_outlets[x->x_n], s, argc, argv);
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_router_new(t_symbol *s, int argc, t_atom *argv) {
    t_alien_router *x = (t_alien_router *)pd_new(alien_router_class);

    // Store the number of routing symbols
    x->x_n = argc;

    // Allocate arrays for symbols and outlets
    if (argc > 0) {
        x->x_vec = (t_symbol **)getbytes(argc * sizeof(t_symbol *));
    } else {
        x->x_vec = NULL;
    }
    x->x_outlets = (t_outlet **)getbytes((argc + 1) * sizeof(t_outlet *));

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

    // Create outlets: one for each routing symbol, plus one for unmatched
    for (int i = 0; i <= argc; i++) {
        x->x_outlets[i] = outlet_new(&x->x_obj, &s_list);
    }

    return (void *)x;
}

static void alien_router_free(t_alien_router *x) {
    if (x->x_n > 0) {
        freebytes(x->x_vec, x->x_n * sizeof(t_symbol *));
    }
    freebytes(x->x_outlets, (x->x_n + 1) * sizeof(t_outlet *));
}

// ============================================================================
// SETUP
// ============================================================================

void alien_router_setup(void) {
    alien_router_class = class_new(gensym("alien_router"),
        (t_newmethod)alien_router_new,
        (t_method)alien_router_free,
        sizeof(t_alien_router),
        CLASS_DEFAULT,
        A_GIMME,
        0);
    class_addanything(alien_router_class, alien_router_anything);
}
