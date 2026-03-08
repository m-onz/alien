/*
 * alien_entangled - Pattern language with global clear signal
 *
 * Like [alien] but listens for a global "clear" broadcast from
 * [alien_router_entangled]. When clear is received, outputs (seq -)
 * to reset the downstream sequencer.
 *
 * Usage: [alien_entangled]
 * Input: symbol with lisp expression, e.g., "(seq 1 2 3)" or "(euclid 5 8)"
 * Output: list of numbers and hyphens (hyphens output as -1)
 *
 * Special behavior:
 *   - Binds to global "__alien_entangle__" symbol
 *   - When "clear" message received, outputs single "-" (rest)
 *   - Works with alien_router_entangled for atomic section changes
 */

#include "alien_core.h"

static t_class *alien_entangled_class;

// Global symbol for entanglement communication
static t_symbol *s_entangle = NULL;

typedef struct _alien_entangled {
    t_object x_obj;
    t_outlet *x_out;
} t_alien_entangled;

// ============================================================================
// CLEAR HANDLER - Called when alien_router_entangled broadcasts clear
// ============================================================================

static void alien_entangled_clear(t_alien_entangled *x) {
    // Output a single rest to clear the downstream sequencer
    t_atom rest;
    SETSYMBOL(&rest, gensym("-"));
    outlet_list(x->x_out, &s_list, 1, &rest);
}

// ============================================================================
// PATTERN EVALUATION - Same as alien.c
// ============================================================================

#define ALIEN_MAX_INPUT 16384
#define ALIEN_MAX_TOKENS 4096

static void alien_entangled_anything(t_alien_entangled *x, t_symbol *s, int argc, t_atom *argv) {
    // Concatenate all atoms into a single string
    char input[ALIEN_MAX_INPUT];
    char *p = input;
    size_t remaining = sizeof(input) - 1;
    int truncated = 0;

    // Add the selector (first symbol)
    int len = snprintf(p, remaining, "%s", s->s_name);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= len;
    } else if (len > 0) {
        truncated = 1;
    }

    // Add all arguments
    for (int i = 0; i < argc && remaining > 1 && !truncated; i++) {
        if (argv[i].a_type == A_FLOAT) {
            float f = atom_getfloat(&argv[i]);
            if (f == (int)f) {
                len = snprintf(p, remaining, " %d", (int)f);
            } else {
                len = snprintf(p, remaining, " %g", f);
            }
        } else if (argv[i].a_type == A_SYMBOL) {
            len = snprintf(p, remaining, " %s", atom_getsymbol(&argv[i])->s_name);
        } else {
            continue;
        }

        if (len > 0 && (size_t)len < remaining) {
            p += len;
            remaining -= len;
        } else if (len > 0) {
            truncated = 1;
        }
    }
    *p = '\0';

    if (truncated) {
        pd_error(x, "alien_entangled: input truncated (max %d chars)", ALIEN_MAX_INPUT);
    }

    // Tokenize
    Token tokens[ALIEN_MAX_TOKENS];
    int token_count = tokenize(input, tokens, ALIEN_MAX_TOKENS);
    if (token_count < 0) {
        pd_error(x, "alien_entangled: %s", g_error_message);
        return;
    }

    // Parse
    ASTNode *ast = parse(tokens, token_count);
    if (!ast) {
        pd_error(x, "alien_entangled: %s", g_error_message);
        return;
    }

    // Evaluate
    Sequence *result = eval_node(ast);
    if (!result) {
        pd_error(x, "alien_entangled: %s", g_error_message);
        ast_free(ast);
        return;
    }

    // Output as PD list (convert -1 to symbol "-")
    if (result->length > 0) {
        t_atom *out_list = (t_atom*)getbytes(sizeof(t_atom) * result->length);
        for (int i = 0; i < result->length; i++) {
            if (result->values[i] == -1) {
                SETSYMBOL(&out_list[i], gensym("-"));
            } else {
                SETFLOAT(&out_list[i], (t_float)result->values[i]);
            }
        }
        outlet_list(x->x_out, &s_list, result->length, out_list);
        freebytes(out_list, sizeof(t_atom) * result->length);
    }

    seq_free(result);
    ast_free(ast);
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_entangled_new(void) {
    t_alien_entangled *x = (t_alien_entangled *)pd_new(alien_entangled_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    
    // Bind to global entanglement symbol to receive clear messages
    pd_bind(&x->x_obj.ob_pd, s_entangle);
    
    return (void *)x;
}

static void alien_entangled_free(t_alien_entangled *x) {
    // Unbind from global entanglement symbol
    pd_unbind(&x->x_obj.ob_pd, s_entangle);
}

// ============================================================================
// SETUP
// ============================================================================

void alien_entangled_setup(void) {
    // Initialize global entanglement symbol
    s_entangle = gensym("__alien_entangle__");
    
    alien_entangled_class = class_new(gensym("alien_entangled"),
        (t_newmethod)alien_entangled_new,
        (t_method)alien_entangled_free,
        sizeof(t_alien_entangled),
        CLASS_DEFAULT,
        0);
    class_addanything(alien_entangled_class, alien_entangled_anything);
    class_addmethod(alien_entangled_class, (t_method)alien_entangled_clear,
                    gensym("clear"), 0);
    post("alien_entangled %s - pattern language with global sync", ALIEN_VERSION_STRING);
}
