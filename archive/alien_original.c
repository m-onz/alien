/*
 * alien - Lisp-like pattern language for Pure Data
 * Generates sequences for else/sequencer and other PD objects
 *
 * Usage: [alien]
 * Input: symbol with lisp expression, e.g., "(seq 1 2 3)" or "(euclid 5 8)"
 * Output: list of numbers and hyphens (hyphens output as -1)
 */

// PD is defined by the Makefile (-DPD), which enables Pure Data memory functions in alien_core.h
#include "alien_core.h"

static t_class *alien_class;

typedef struct _alien {
    t_object x_obj;
    t_outlet *x_out;
} t_alien;

// ============================================================================
// PURE DATA INTERFACE
// ============================================================================

// Maximum sizes for input processing
#define ALIEN_MAX_INPUT 16384
#define ALIEN_MAX_TOKENS 4096

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
            // Check if it's an integer
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

static void *alien_new(void) {
    t_alien *x = (t_alien *)pd_new(alien_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    return (void *)x;
}

void alien_setup(void) {
    alien_class = class_new(gensym("alien"),
        (t_newmethod)alien_new,
        0,
        sizeof(t_alien),
        CLASS_DEFAULT,
        0);
    class_addanything(alien_class, alien_anything);
    post("alien %s - lisp-like pattern language", ALIEN_VERSION_STRING);
}
