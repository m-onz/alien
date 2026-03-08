/*
 * alien - Unified pattern language for Pure Data
 *
 * A single object that handles pattern generation, buffering, and global sync.
 *
 * Usage:
 *   [alien]           - Anonymous, immediate mode
 *   [alien kick]      - Named "kick", can receive global messages
 *
 * Direct input (both modes):
 *   (seq 1 2 3)       - Evaluates immediately, outputs pattern
 *
 * Global messages (named mode only):
 *   Send to [s kick]: (euclid 4 16) - Buffers pattern in "kick" alien
 *   Send to [s alien_sync]: bang   - All named aliens: clear + apply buffer
 *   Send to [s alien_clear]: bang  - All named aliens: clear to rest
 *
 * Output: list of numbers and hyphens (hyphens output as -1)
 */

#include "alien_core.h"

static t_class *alien_class;

// Global symbols for sync/clear
static t_symbol *s_alien_sync = NULL;
static t_symbol *s_alien_clear = NULL;

#define ALIEN_MAX_INPUT 16384
#define ALIEN_MAX_TOKENS 4096
#define ALIEN_MAX_BUFFER 4096

typedef struct _alien {
    t_object x_obj;
    t_outlet *x_out;
    t_symbol *x_name;           // NULL for anonymous, symbol for named
    char x_buffer[ALIEN_MAX_BUFFER];  // buffered pattern string
    int x_has_buffer;           // 1 if buffer has pending pattern
    int x_is_global_msg;        // flag to track if current message is global
} t_alien;

// ============================================================================
// PATTERN EVALUATION
// ============================================================================

static void alien_evaluate_and_output(t_alien *x, const char *input) {
    Token tokens[ALIEN_MAX_TOKENS];
    int token_count = tokenize(input, tokens, ALIEN_MAX_TOKENS);
    if (token_count < 0) {
        pd_error(x, "alien: %s", g_error_message);
        return;
    }

    ASTNode *ast = parse(tokens, token_count);
    if (!ast) {
        pd_error(x, "alien: %s", g_error_message);
        return;
    }

    Sequence *result = eval_node(ast);
    if (!result) {
        pd_error(x, "alien: %s", g_error_message);
        ast_free(ast);
        return;
    }

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

static void alien_output_rest(t_alien *x) {
    t_atom rest;
    SETSYMBOL(&rest, gensym("-"));
    outlet_list(x->x_out, &s_list, 1, &rest);
}

// ============================================================================
// SYNC HANDLER - Called via [s alien_sync]
// ============================================================================

static void alien_do_sync(t_alien *x) {
    // Only named aliens respond to sync
    if (!x->x_name) return;
    
    // Clear current pattern
    alien_output_rest(x);
    
    // If we have a buffered pattern, evaluate and output it
    if (x->x_has_buffer && x->x_buffer[0] != '\0') {
        alien_evaluate_and_output(x, x->x_buffer);
    }
    x->x_has_buffer = 0;
    x->x_buffer[0] = '\0';
}

// ============================================================================
// CLEAR HANDLER - Called via [s alien_clear]
// ============================================================================

static void alien_do_clear(t_alien *x) {
    // Only named aliens respond to clear
    if (!x->x_name) return;
    
    alien_output_rest(x);
    x->x_has_buffer = 0;
    x->x_buffer[0] = '\0';
}

// ============================================================================
// BANG HANDLER - For sync/clear symbols
// ============================================================================

static void alien_bang(t_alien *x) {
    // Bang does nothing for anonymous aliens
    // For named aliens, bang is used for sync (when received via alien_sync)
    // This is handled by checking which symbol the message came from
    // But Pd doesn't tell us that, so we use a different approach:
    // We'll use "sync" and "clear" as explicit messages instead
}

// ============================================================================
// SYNC/CLEAR MESSAGE HANDLERS
// ============================================================================

static void alien_sync_msg(t_alien *x) {
    alien_do_sync(x);
}

static void alien_clear_msg(t_alien *x) {
    alien_do_clear(x);
}

// ============================================================================
// BUFFER PATTERN - Store for later sync
// ============================================================================

static void alien_buffer_pattern(t_alien *x, const char *input) {
    strncpy(x->x_buffer, input, ALIEN_MAX_BUFFER - 1);
    x->x_buffer[ALIEN_MAX_BUFFER - 1] = '\0';
    x->x_has_buffer = 1;
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

static void alien_anything(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    // Build input string
    char input[ALIEN_MAX_INPUT];
    char *p = input;
    size_t remaining = sizeof(input) - 1;
    int truncated = 0;

    int len = snprintf(p, remaining, "%s", s->s_name);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= len;
    } else if (len > 0) {
        truncated = 1;
    }

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

    // Always evaluate immediately for direct input
    alien_evaluate_and_output(x, input);
}

// ============================================================================
// GLOBAL PATTERN RECEIVER - For named aliens receiving via [s name]
// ============================================================================

static void alien_list(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    
    // Build input string from list
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

    // If named, buffer for sync; if anonymous, evaluate immediately
    if (x->x_name) {
        alien_buffer_pattern(x, input);
    } else {
        alien_evaluate_and_output(x, input);
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
    x->x_is_global_msg = 0;
    
    if (name && name != &s_ && strlen(name->s_name) > 0) {
        x->x_name = name;
        
        // Bind to our name symbol to receive patterns via [s name]
        pd_bind(&x->x_obj.ob_pd, x->x_name);
        
        // Bind to global sync symbol
        pd_bind(&x->x_obj.ob_pd, s_alien_sync);
        
        // Bind to global clear symbol
        pd_bind(&x->x_obj.ob_pd, s_alien_clear);
        
        post("alien: '%s' registered (use [s %s] to send patterns, [s alien_sync] to sync)", 
             x->x_name->s_name, x->x_name->s_name);
    } else {
        x->x_name = NULL;
    }
    
    return (void *)x;
}

static void alien_free(t_alien *x) {
    if (x->x_name) {
        pd_unbind(&x->x_obj.ob_pd, x->x_name);
        pd_unbind(&x->x_obj.ob_pd, s_alien_sync);
        pd_unbind(&x->x_obj.ob_pd, s_alien_clear);
    }
}

// ============================================================================
// SETUP
// ============================================================================

void alien_setup(void) {
    s_alien_sync = gensym("alien_sync");
    s_alien_clear = gensym("alien_clear");
    
    alien_class = class_new(gensym("alien"),
        (t_newmethod)alien_new,
        (t_method)alien_free,
        sizeof(t_alien),
        CLASS_DEFAULT,
        A_DEFSYM,
        0);
    
    class_addanything(alien_class, alien_anything);
    class_addlist(alien_class, alien_list);
    class_addbang(alien_class, alien_bang);
    
    // Explicit sync/clear messages (work both directly and via [s alien_sync])
    class_addmethod(alien_class, (t_method)alien_sync_msg, gensym("sync"), 0);
    class_addmethod(alien_class, (t_method)alien_clear_msg, gensym("clear"), 0);
    
    post("alien %s - lisp-like pattern language", ALIEN_VERSION_STRING);
    post("  [alien] for immediate mode");
    post("  [alien name] for named mode with global sync");
    post("  Send patterns via [s name], sync via [s alien_sync]");
}
