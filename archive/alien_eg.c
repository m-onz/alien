/*
 * alien_eg - Entangled Group pattern language
 *
 * Like [alien_entangled] but with a namespace/group argument.
 * Only responds to clear signals from [alien_eg_router] with matching group.
 *
 * Usage: [alien_eg drums]
 *        [alien_eg synths]
 *        [alien_eg section1]
 *
 * Input: symbol with lisp expression, e.g., "(seq 1 2 3)" or "(euclid 5 8)"
 * Output: list of numbers and hyphens (hyphens output as -1)
 *
 * Special behavior:
 *   - Binds to "__alien_eg_<group>__" symbol
 *   - Only clears when matching group router flushes
 *   - Multiple groups can coexist independently
 */

#include "alien_core.h"

static t_class *alien_eg_class;

typedef struct _alien_eg {
    t_object x_obj;
    t_outlet *x_out;
    t_symbol *x_group;      // group name
    t_symbol *x_bind_sym;   // computed bind symbol "__alien_eg_<group>__"
} t_alien_eg;

// ============================================================================
// CLEAR HANDLER - Called when matching alien_eg_router broadcasts clear
// ============================================================================

static void alien_eg_clear(t_alien_eg *x) {
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

static void alien_eg_anything(t_alien_eg *x, t_symbol *s, int argc, t_atom *argv) {
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
        pd_error(x, "alien_eg: input truncated (max %d chars)", ALIEN_MAX_INPUT);
    }

    // Tokenize
    Token tokens[ALIEN_MAX_TOKENS];
    int token_count = tokenize(input, tokens, ALIEN_MAX_TOKENS);
    if (token_count < 0) {
        pd_error(x, "alien_eg: %s", g_error_message);
        return;
    }

    // Parse
    ASTNode *ast = parse(tokens, token_count);
    if (!ast) {
        pd_error(x, "alien_eg: %s", g_error_message);
        return;
    }

    // Evaluate
    Sequence *result = eval_node(ast);
    if (!result) {
        pd_error(x, "alien_eg: %s", g_error_message);
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

static void *alien_eg_new(t_symbol *group) {
    t_alien_eg *x = (t_alien_eg *)pd_new(alien_eg_class);
    x->x_out = outlet_new(&x->x_obj, &s_list);
    
    // Store group name (default to "default" if not specified)
    if (group && group != &s_) {
        x->x_group = group;
    } else {
        x->x_group = gensym("default");
    }
    
    // Create bind symbol: "__alien_eg_<group>__"
    char bind_name[256];
    snprintf(bind_name, sizeof(bind_name), "__alien_eg_%s__", x->x_group->s_name);
    x->x_bind_sym = gensym(bind_name);
    
    // Bind to group-specific symbol to receive clear messages
    pd_bind(&x->x_obj.ob_pd, x->x_bind_sym);
    
    return (void *)x;
}

static void alien_eg_free(t_alien_eg *x) {
    // Unbind from group-specific symbol
    pd_unbind(&x->x_obj.ob_pd, x->x_bind_sym);
}

// ============================================================================
// SETUP
// ============================================================================

void alien_eg_setup(void) {
    alien_eg_class = class_new(gensym("alien_eg"),
        (t_newmethod)alien_eg_new,
        (t_method)alien_eg_free,
        sizeof(t_alien_eg),
        CLASS_DEFAULT,
        A_DEFSYM,
        0);
    class_addanything(alien_eg_class, alien_eg_anything);
    class_addmethod(alien_eg_class, (t_method)alien_eg_clear,
                    gensym("clear"), 0);
    post("alien_eg %s - entangled group pattern language", ALIEN_VERSION_STRING);
}
