/*
 * alien - Entangled pattern evaluator for Pure Data
 *
 * Two modes:
 *   [alien]              - generator mode: expression in, list out
 *   [alien kick]         - named mode: receives via [s kick], outputs list
 *   [alien kick -unsync] - named mode: independent (no sync batching)
 *
 * Named instances share a sync clock. Pattern changes arriving within
 * 100ms are batched — all entangled instances output their new patterns
 * simultaneously. Feed the output to [else/sequencer] for stepping.
 */

#include "alien_core.h"

static t_class *alien_class;

#define ALIEN_MAX_STEPS 1024
#define ALIEN_SYNC_DELAY_MS 100.0
#define ALIEN_GROUP_INIT_CAP 16

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

typedef struct _alien t_alien;
typedef struct _alien_transport t_alien_transport;
static void alien_sync_tick(t_alien *x);
static void transport_do_pending_reset(t_alien_transport *x);

// ============================================================================
// GLOBAL ENTANGLED GROUP
// ============================================================================

static struct {
    t_alien **instances;
    int count;
    int capacity;
    t_alien_transport **transports;
    int transport_count;
    int transport_capacity;
    t_clock *sync_clock;
    t_alien *clock_owner;
} g_group = {NULL, 0, 0, NULL, 0, 0, NULL, NULL};

static void group_register(t_alien *x) {
    if (g_group.instances == NULL) {
        g_group.capacity = ALIEN_GROUP_INIT_CAP;
        g_group.instances = (t_alien **)getbytes(sizeof(t_alien *) * g_group.capacity);
        g_group.count = 0;
    }
    if (g_group.count >= g_group.capacity) {
        int old_cap = g_group.capacity;
        g_group.capacity *= 2;
        g_group.instances = (t_alien **)resizebytes(g_group.instances,
            sizeof(t_alien *) * old_cap,
            sizeof(t_alien *) * g_group.capacity);
    }
    g_group.instances[g_group.count++] = x;

    if (!g_group.sync_clock) {
        g_group.sync_clock = clock_new(x, (t_method)alien_sync_tick);
        g_group.clock_owner = x;
    }
}

static void group_unregister(t_alien *x);  // implemented after struct definition

static void group_register_transport(t_alien_transport *x) {
    if (g_group.transports == NULL) {
        g_group.transport_capacity = ALIEN_GROUP_INIT_CAP;
        g_group.transports = (t_alien_transport **)getbytes(
            sizeof(t_alien_transport *) * g_group.transport_capacity);
        g_group.transport_count = 0;
    }
    if (g_group.transport_count >= g_group.transport_capacity) {
        int old_cap = g_group.transport_capacity;
        g_group.transport_capacity *= 2;
        g_group.transports = (t_alien_transport **)resizebytes(g_group.transports,
            sizeof(t_alien_transport *) * old_cap,
            sizeof(t_alien_transport *) * g_group.transport_capacity);
    }
    g_group.transports[g_group.transport_count++] = x;
}

static void group_unregister_transport(t_alien_transport *x) {
    for (int i = 0; i < g_group.transport_count; i++) {
        if (g_group.transports[i] == x) {
            g_group.transports[i] = g_group.transports[--g_group.transport_count];
            break;
        }
    }
}

// ============================================================================
// PER-INSTANCE STRUCT
// ============================================================================

struct _alien {
    t_object x_obj;

    // Shared
    t_outlet *x_list_out;       // list outlet (both modes)
    t_symbol *x_name;           // NULL = generator mode

    // Named mode
    t_atom x_pattern[ALIEN_MAX_STEPS];
    int x_length;
    char x_buffer[4096];
    int x_has_buffer;
    int x_entangled;            // 1 = in group, 0 = independent
    t_clock *x_local_clock;     // for unentangled pattern sync
};

// ============================================================================
// DEFERRED: group_unregister (needs complete t_alien definition)
// ============================================================================

static void group_unregister(t_alien *x) {
    for (int i = 0; i < g_group.count; i++) {
        if (g_group.instances[i] == x) {
            g_group.instances[i] = g_group.instances[--g_group.count];
            break;
        }
    }
    if (g_group.clock_owner == x) {
        if (g_group.count > 0) {
            // Check if any instances have buffered patterns pending
            int has_pending = 0;
            for (int i = 0; i < g_group.count; i++) {
                if (g_group.instances[i]->x_has_buffer) { has_pending = 1; break; }
            }
            g_group.clock_owner = g_group.instances[0];
            t_clock *old = g_group.sync_clock;
            g_group.sync_clock = clock_new(g_group.clock_owner, (t_method)alien_sync_tick);
            clock_free(old);
            // Reschedule if patterns were waiting on the old clock
            if (has_pending) {
                clock_delay(g_group.sync_clock, ALIEN_SYNC_DELAY_MS);
            }
        } else {
            clock_free(g_group.sync_clock);
            g_group.sync_clock = NULL;
            g_group.clock_owner = NULL;
        }
    }
}

// ============================================================================
// PATTERN EVALUATION
// ============================================================================

#define ALIEN_MAX_TOKENS 4096

static int alien_evaluate_to_atoms(t_alien *x, const char *input, t_atom *out, int max_len) {
    Token *tokens = (Token *)getbytes(sizeof(Token) * ALIEN_MAX_TOKENS);
    if (!tokens) {
        pd_error(x, "alien: memory allocation failed for tokens");
        return 0;
    }

    int token_count = tokenize(input, tokens, ALIEN_MAX_TOKENS);
    if (token_count < 0) {
        pd_error(x, "alien: %s", g_error_message);
        freebytes(tokens, sizeof(Token) * ALIEN_MAX_TOKENS);
        return 0;
    }

    ASTNode *ast = parse(tokens, token_count);
    freebytes(tokens, sizeof(Token) * ALIEN_MAX_TOKENS);
    if (!ast) {
        pd_error(x, "alien: %s", g_error_message);
        return 0;
    }

    Sequence *result = eval_node(ast);
    if (!result) {
        pd_error(x, "alien: %s", g_error_message);
        ast_free(ast);
        return 0;
    }

    int len = (result->length < max_len) ? result->length : max_len;
    for (int i = 0; i < len; i++) {
        if (result->values[i] == ALIEN_REST) {
            SETSYMBOL(&out[i], gensym("-"));
        } else {
            SETFLOAT(&out[i], (t_float)result->values[i]);
        }
    }

    int final_len = len;
    seq_free(result);
    ast_free(ast);
    return final_len;
}

// ============================================================================
// OUTPUT PATTERN AS LIST
// ============================================================================

static void alien_output_pattern(t_alien *x) {
    if (x->x_length > 0) {
        outlet_list(x->x_list_out, &s_list, x->x_length, x->x_pattern);
    }
}

// ============================================================================
// GENERATOR MODE
// ============================================================================

static void alien_anything_generator(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    #define ALIEN_INPUT_BUF 16384
    char *input = (char *)getbytes(ALIEN_INPUT_BUF);
    t_atom *out = (t_atom *)getbytes(sizeof(t_atom) * ALIEN_MAX_STEPS);
    if (!input || !out) {
        pd_error(x, "alien: memory allocation failed");
        if (input) freebytes(input, ALIEN_INPUT_BUF);
        if (out) freebytes(out, sizeof(t_atom) * ALIEN_MAX_STEPS);
        return;
    }

    char *p = input;
    size_t remaining = ALIEN_INPUT_BUF - 1;

    int len = snprintf(p, remaining, "%s", s->s_name);
    if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }

    for (int i = 0; i < argc && remaining > 1; i++) {
        if (argv[i].a_type == A_FLOAT) {
            float f = atom_getfloat(&argv[i]);
            if (f == (int)f) len = snprintf(p, remaining, " %d", (int)f);
            else len = snprintf(p, remaining, " %g", f);
        } else if (argv[i].a_type == A_SYMBOL) {
            len = snprintf(p, remaining, " %s", atom_getsymbol(&argv[i])->s_name);
        } else continue;
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }
    }
    *p = '\0';

    int out_len = alien_evaluate_to_atoms(x, input, out, ALIEN_MAX_STEPS);
    if (out_len > 0) {
        outlet_list(x->x_list_out, &s_list, out_len, out);
    }

    freebytes(input, ALIEN_INPUT_BUF);
    freebytes(out, sizeof(t_atom) * ALIEN_MAX_STEPS);
    #undef ALIEN_INPUT_BUF
}

// ============================================================================
// BANG — re-output current pattern
// ============================================================================

static void alien_bang(t_alien *x) {
    if (!x->x_name) return;
    alien_output_pattern(x);
}

// ============================================================================
// PATTERN INPUT (via pd_bind to name)
// ============================================================================

static void alien_buffer_pattern(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    char *p = x->x_buffer;
    size_t remaining = sizeof(x->x_buffer) - 1;

    // Include selector if it's not a Pd built-in
    if (s && s != &s_list && s != &s_symbol && s != &s_float && s != &s_bang) {
        int len = snprintf(p, remaining, "%s", s->s_name);
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }
        if (argc > 0) {
            int len2 = snprintf(p, remaining, " ");
            if (len2 > 0 && (size_t)len2 < remaining) { p += len2; remaining -= len2; }
        }
    }

    for (int i = 0; i < argc && remaining > 1; i++) {
        int len = 0;
        if (argv[i].a_type == A_FLOAT) {
            float f = atom_getfloat(&argv[i]);
            if (f == (int)f) len = snprintf(p, remaining, "%s%d", (i > 0 ? " " : ""), (int)f);
            else len = snprintf(p, remaining, "%s%g", (i > 0 ? " " : ""), f);
        } else if (argv[i].a_type == A_SYMBOL) {
            len = snprintf(p, remaining, "%s%s", (i > 0 ? " " : ""), atom_getsymbol(&argv[i])->s_name);
        }
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= len; }
    }
    *p = '\0';
    x->x_has_buffer = 1;

    // Schedule sync
    if (x->x_entangled && g_group.sync_clock) {
        clock_delay(g_group.sync_clock, ALIEN_SYNC_DELAY_MS);
    } else {
        clock_delay(x->x_local_clock, ALIEN_SYNC_DELAY_MS);
    }
}

// ============================================================================
// SYNC TICKS — evaluate buffered patterns, output lists
// ============================================================================

static void alien_sync_tick(t_alien *x) {
    (void)x;

    // Evaluate ALL buffered group patterns
    for (int i = 0; i < g_group.count; i++) {
        t_alien *inst = g_group.instances[i];
        if (inst->x_has_buffer && inst->x_buffer[0] != '\0') {
            inst->x_length = alien_evaluate_to_atoms(inst, inst->x_buffer,
                inst->x_pattern, ALIEN_MAX_STEPS);
            inst->x_has_buffer = 0;
            inst->x_buffer[0] = '\0';
        }
    }

    // Reset transports in the same tick as pattern output
    for (int i = 0; i < g_group.transport_count; i++)
        transport_do_pending_reset(g_group.transports[i]);

    // Output all group patterns together
    for (int i = 0; i < g_group.count; i++) {
        alien_output_pattern(g_group.instances[i]);
    }
}

static void alien_local_tick(t_alien *x) {
    if (x->x_has_buffer && x->x_buffer[0] != '\0') {
        x->x_length = alien_evaluate_to_atoms(x, x->x_buffer,
            x->x_pattern, ALIEN_MAX_STEPS);
        x->x_has_buffer = 0;
        x->x_buffer[0] = '\0';
    }
    alien_output_pattern(x);
}

// ============================================================================
// MESSAGES
// ============================================================================

static void alien_unsync(t_alien *x) {
    if (!x->x_name) return;
    if (x->x_entangled) {
        group_unregister(x);
        x->x_entangled = 0;
    }
}

static void alien_sync(t_alien *x) {
    if (!x->x_name) return;
    if (!x->x_entangled) {
        x->x_entangled = 1;
        group_register(x);
    }
}

static void alien_reset(t_alien *x) {
    if (!x->x_name) return;
    x->x_length = 0;
}

// ============================================================================
// ANYTHING / LIST HANDLERS
// ============================================================================

static void alien_anything(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    if (!x->x_name) {
        alien_anything_generator(x, s, argc, argv);
    } else {
        alien_buffer_pattern(x, s, argc, argv);
    }
}

static void alien_list(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (!x->x_name) {
        // Generator mode: treat as pattern input
        if (argc > 0 && argv[0].a_type == A_SYMBOL) {
            alien_anything_generator(x, atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
        }
        return;
    }
    // Named mode: ignore float-led lists (likely step output from sequencers).
    // Patterns always arrive as symbol-led messages like (seq 1 2 3)
    // which go through alien_anything, not here.
}

static void alien_float(t_alien *x, t_floatarg f) {
    (void)x; (void)f;
    // Ignore floats in both modes. Named aliens only accept
    // symbol-led pattern expressions like (seq 1 2 3).
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien *x = (t_alien *)pd_new(alien_class);

    x->x_name = NULL;
    x->x_list_out = NULL;
    x->x_length = 0;
    x->x_has_buffer = 0;
    x->x_buffer[0] = '\0';
    x->x_entangled = 0;
    x->x_local_clock = clock_new(x, (t_method)alien_local_tick);

    // Always create the outlet so the object is safe even on error paths
    x->x_list_out = outlet_new(&x->x_obj, &s_list);

    if (argc == 0) {
        // Generator mode: expression in, list out
    } else {
        // Named mode
        if (argv[0].a_type == A_SYMBOL) {
            x->x_name = atom_getsymbol(&argv[0]);
        } else {
            pd_error(x, "alien: first argument must be a name");
            return (void *)x;
        }

        // Check for -unsync flag
        int unsync = 0;
        for (int i = 1; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL &&
                strcmp(atom_getsymbol(&argv[i])->s_name, "-unsync") == 0) {
                unsync = 1;
            }
        }

        // Bind to own name and shared "alien" for broadcast
        pd_bind(&x->x_obj.ob_pd, x->x_name);
        pd_bind(&x->x_obj.ob_pd, gensym("all"));

        if (!unsync) {
            x->x_entangled = 1;
            group_register(x);
        }
    }

    return (void *)x;
}

static void alien_free(t_alien *x) {
    clock_free(x->x_local_clock);
    if (x->x_name) {
        pd_unbind(&x->x_obj.ob_pd, x->x_name);
        pd_unbind(&x->x_obj.ob_pd, gensym("all"));
        if (x->x_entangled) {
            group_unregister(x);
        }
    }
}

// ============================================================================
// ALIEN_TRANSPORT — clock divider with synchronized reset
// ============================================================================

static t_class *alien_transport_class;

struct _alien_transport {
    t_object x_obj;
    unsigned int x_count;
    int x_frozen;           // suppress outlets while waiting for sync
    t_outlet *x_out[5];     // 0:raw, 1:/2, 2:/4, 3:/8, 4:/16
};

// Called by alien_sync_tick — unfreeze and reset at the exact moment
// patterns are output, so sequencers get new patterns + fresh count together
static void transport_do_pending_reset(t_alien_transport *x) {
    if (x->x_frozen) {
        x->x_count = 0;
        x->x_frozen = 0;
    }
}

static void transport_reset(t_alien_transport *x) {
    x->x_count = 0;
    x->x_frozen = 0;
}

static void transport_bang(t_alien_transport *x) {
    if (x->x_frozen) return; // suppress ticks while waiting for patterns

    unsigned int c = x->x_count++;

    // Fire right-to-left (Pd outlet convention)
    if ((c % 16) == 0) outlet_bang(x->x_out[4]);
    if ((c % 8) == 0)  outlet_bang(x->x_out[3]);
    if ((c % 4) == 0)  outlet_bang(x->x_out[2]);
    if ((c % 2) == 0)  outlet_bang(x->x_out[1]);
    outlet_bang(x->x_out[0]);
}

// Pattern messages arriving via "all" — freeze until sync_tick fires
static void transport_anything(t_alien_transport *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; (void)argc; (void)argv;
    x->x_frozen = 1;
    if (g_group.sync_clock)
        clock_delay(g_group.sync_clock, ALIEN_SYNC_DELAY_MS);
}

static void *transport_new(void) {
    t_alien_transport *x = (t_alien_transport *)pd_new(alien_transport_class);
    x->x_count = 0;
    x->x_frozen = 0;

    // Inlet 1: reset
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_bang, gensym("reset"));

    // 5 outlets: raw, /2, /4, /8, /16
    for (int i = 0; i < 5; i++)
        x->x_out[i] = outlet_new(&x->x_obj, &s_bang);

    // Listen to "all" for synchronized reset
    pd_bind(&x->x_obj.ob_pd, gensym("all"));
    group_register_transport(x);

    return (void *)x;
}

static void transport_free(t_alien_transport *x) {
    pd_unbind(&x->x_obj.ob_pd, gensym("all"));
    group_unregister_transport(x);
}

// ============================================================================
// SETUP
// ============================================================================

void alien_setup(void) {
    alien_class = class_new(gensym("alien"),
        (t_newmethod)alien_new,
        (t_method)alien_free,
        sizeof(t_alien),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addbang(alien_class, alien_bang);
    class_addfloat(alien_class, alien_float);
    class_addlist(alien_class, alien_list);
    class_addanything(alien_class, alien_anything);
    class_addmethod(alien_class, (t_method)alien_reset, gensym("reset"), 0);
    class_addmethod(alien_class, (t_method)alien_unsync, gensym("unsync"), 0);
    class_addmethod(alien_class, (t_method)alien_sync, gensym("sync"), 0);

    // alien_transport class
    alien_transport_class = class_new(gensym("alien_transport"),
        (t_newmethod)transport_new,
        (t_method)transport_free,
        sizeof(t_alien_transport),
        CLASS_DEFAULT,
        0);

    class_addbang(alien_transport_class, transport_bang);
    class_addanything(alien_transport_class, transport_anything);
    class_addmethod(alien_transport_class, (t_method)transport_reset,
        gensym("reset"), 0);

    post("alien %s - entangled pattern evaluator", ALIEN_VERSION_STRING);
}
