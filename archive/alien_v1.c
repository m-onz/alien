/*
 * alien - Unified pattern language for Pure Data
 *
 * A single object that handles pattern generation, buffering, and global sync.
 *
 * Usage:
 *   [alien]           - Anonymous, immediate mode (current behavior)
 *   [alien kick]      - Named "kick", buffers patterns, syncs globally
 *   [alien snare]     - Named "snare", buffers patterns, syncs globally
 *
 * Named mode:
 *   - Receives patterns via global messages: [kick (euclid 4 16)(
 *   - Buffers pattern until "sync" message received
 *   - On sync: clears current pattern, evaluates buffered, outputs
 *
 * Global messages (sent to any named alien, affects all):
 *   sync              - All named aliens: clear + apply buffer + output
 *   sync kick snare   - Only specified aliens sync
 *   clear             - All named aliens: clear to rest pattern
 *
 * Output: list of numbers and hyphens (hyphens output as -1)
 */

#include "alien_core.h"

static t_class *alien_class;

// Global sync symbol - all named aliens bind to this
static t_symbol *s_alien_sync = NULL;

#define ALIEN_MAX_INPUT 16384
#define ALIEN_MAX_TOKENS 4096
#define ALIEN_MAX_BUFFER 4096

typedef struct _alien {
    t_object x_obj;
    t_outlet *x_out;
    t_symbol *x_name;           // NULL for anonymous, symbol for named
    char x_buffer[ALIEN_MAX_BUFFER];  // buffered pattern string
    int x_has_buffer;           // 1 if buffer has pending pattern
} t_alien;

// ============================================================================
// PATTERN EVALUATION
// ============================================================================

static void alien_evaluate_and_output(t_alien *x, const char *input) {
    // Tokenize
    Token tokens[ALIEN_MAX_TOKENS];
    int token_count = tokenize(input, tokens, ALIEN_MAX_TOKENS);
    if (token_count < 0) {
        pd_error(x, "alien: %s", g_error_message);
        return;
    }

    // Parse
    ASTNode *ast = parse(tokens, token_count);
    if (!ast) {
        pd_error(x, "alien: %s", g_error_message);
        return;
    }

    // Evaluate
    Sequence *result = eval_node(ast);
    if (!result) {
        pd_error(x, "alien: %s", g_error_message);
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
// CLEAR - Output rest pattern
// ============================================================================

static void alien_clear(t_alien *x) {
    // Output a single rest
    t_atom rest;
    SETSYMBOL(&rest, gensym("-"));
    outlet_list(x->x_out, &s_list, 1, &rest);
    x->x_has_buffer = 0;
}

// ============================================================================
// SYNC - Clear, apply buffer, output
// ============================================================================

static void alien_sync(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    
    // If arguments provided, check if this alien is in the list
    if (argc > 0 && x->x_name) {
        int found = 0;
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL && atom_getsymbol(&argv[i]) == x->x_name) {
                found = 1;
                break;
            }
        }
        if (!found) return;  // Not in sync list, skip
    }
    
    // Clear current pattern
    alien_clear(x);
    
    // If we have a buffered pattern, evaluate and output it
    if (x->x_has_buffer && x->x_buffer[0] != '\0') {
        alien_evaluate_and_output(x, x->x_buffer);
        x->x_has_buffer = 0;
    }
}

// ============================================================================
// GLOBAL CLEAR - Clear all named aliens
// ============================================================================

static void alien_global_clear(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    
    // If arguments provided, check if this alien is in the list
    if (argc > 0 && x->x_name) {
        int found = 0;
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL && atom_getsymbol(&argv[i]) == x->x_name) {
                found = 1;
                break;
            }
        }
        if (!found) return;
    }
    
    alien_clear(x);
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

static void alien_anything(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
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
        pd_error(x, "alien: input truncated (max %d chars)", ALIEN_MAX_INPUT);
    }

    // Always evaluate immediately when receiving direct input
    // (both named and anonymous aliens accept patterns directly)
    alien_evaluate_and_output(x, input);
}

// ============================================================================
// NAMED PATTERN RECEIVER - Receives "kick (pattern)" globally
// ============================================================================

static void alien_named_pattern(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    // This is called when a message like "kick (euclid 4 16)" is sent globally
    // The selector 's' is the name (e.g., "kick")
    // We only respond if it matches our name
    if (x->x_name && s == x->x_name) {
        // Buffer the pattern for later sync (don't evaluate immediately)
        if (argc > 0) {
            char input[ALIEN_MAX_INPUT];
            char *p = input;
            size_t remaining = sizeof(input) - 1;
            
            for (int i = 0; i < argc && remaining > 1; i++) {
                int len = 0;
                if (argv[i].a_type == A_FLOAT) {
                    float f = atom_getfloat(&argv[i]);
                    if (f == (int)f) {
                        len = snprintf(p, remaining, "%s%d", (i > 0 ? " " : ""), (int)f);
                    } else {
                        len = snprintf(p, remaining, "%s%g", (i > 0 ? " " : ""), f);
                    }
                } else if (argv[i].a_type == A_SYMBOL) {
                    len = snprintf(p, remaining, "%s%s", (i > 0 ? " " : ""), atom_getsymbol(&argv[i])->s_name);
                }
                if (len > 0 && (size_t)len < remaining) {
                    p += len;
                    remaining -= len;
                }
            }
            *p = '\0';
            
            strncpy(x->x_buffer, input, ALIEN_MAX_BUFFER - 1);
            x->x_buffer[ALIEN_MAX_BUFFER - 1] = '\0';
            x->x_has_buffer = 1;
        }
    }
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_new(t_symbol *name) {
    t_alien *x = (t_alien *)pd_new(alien_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    x->x_has_buffer = 0;
    x->x_buffer[0] = '\0';
    
    // Check if name provided
    if (name && name != &s_ && strlen(name->s_name) > 0) {
        x->x_name = name;
        
        // Bind to global sync symbol
        pd_bind(&x->x_obj.ob_pd, s_alien_sync);
        
        // Bind to our name symbol to receive patterns globally
        pd_bind(&x->x_obj.ob_pd, x->x_name);
        
        post("alien: '%s' registered", x->x_name->s_name);
    } else {
        x->x_name = NULL;
    }
    
    return (void *)x;
}

static void alien_free(t_alien *x) {
    if (x->x_name) {
        pd_unbind(&x->x_obj.ob_pd, s_alien_sync);
        pd_unbind(&x->x_obj.ob_pd, x->x_name);
    }
}

// ============================================================================
// SETUP
// ============================================================================

void alien_setup(void) {
    // Initialize global sync symbol
    s_alien_sync = gensym("__alien_sync__");
    
    alien_class = class_new(gensym("alien"),
        (t_newmethod)alien_new,
        (t_method)alien_free,
        sizeof(t_alien),
        CLASS_DEFAULT,
        A_DEFSYM,
        0);
    
    class_addanything(alien_class, alien_anything);
    
    // Global sync message
    class_addmethod(alien_class, (t_method)alien_sync,
                    gensym("sync"), A_GIMME, 0);
    
    // Global clear message
    class_addmethod(alien_class, (t_method)alien_global_clear,
                    gensym("clear"), A_GIMME, 0);
    
    post("alien %s - lisp-like pattern language", ALIEN_VERSION_STRING);
}
