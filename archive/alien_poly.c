/*
 * alien - Polymorphic pattern sequencer for Pure Data
 *
 * The simplest possible design:
 *   [alien]                    - pattern generator (outputs list)
 *   [alien kick snare hihat]   - multi-voice sequencer (one outlet per voice)
 *
 * Multi-voice mode:
 *   - Bang advances step, outputs current value on each outlet
 *   - Each voice receives patterns via [s name]
 *   - Patterns auto-sync within 5ms window
 *   - All voices share same step counter (perfect sync)
 *   - Last outlet: bang on loop
 */

#include "alien_core.h"

static t_class *alien_class;

#define ALIEN_MAX_VOICES 32
#define ALIEN_MAX_STEPS 1024
#define ALIEN_SYNC_DELAY_MS 5.0

typedef struct _voice {
    t_symbol *name;
    t_outlet *outlet;
    t_atom pattern[ALIEN_MAX_STEPS];
    int length;
    char buffer[4096];      // pending pattern string
    int has_buffer;
} t_voice;

typedef struct _alien {
    t_object x_obj;
    
    // Pattern generator mode (no voices)
    t_outlet *x_list_out;
    
    // Multi-voice mode
    t_voice x_voices[ALIEN_MAX_VOICES];
    int x_num_voices;
    int x_step;
    t_outlet *x_loop_out;
    t_clock *x_sync_clock;
} t_alien;

// Forward declarations
static void alien_sync_tick(t_alien *x);

// ============================================================================
// PATTERN EVALUATION (shared)
// ============================================================================

static int alien_evaluate_to_atoms(t_alien *x, const char *input, t_atom *out, int max_len) {
    Token tokens[4096];
    int token_count = tokenize(input, tokens, 4096);
    if (token_count < 0) {
        pd_error(x, "alien: %s", g_error_message);
        return 0;
    }

    ASTNode *ast = parse(tokens, token_count);
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
        if (result->values[i] == -1) {
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
// PATTERN GENERATOR MODE
// ============================================================================

static void alien_anything_generator(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    char input[16384];
    char *p = input;
    size_t remaining = sizeof(input) - 1;

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

    t_atom out[ALIEN_MAX_STEPS];
    int out_len = alien_evaluate_to_atoms(x, input, out, ALIEN_MAX_STEPS);
    if (out_len > 0) {
        outlet_list(x->x_list_out, &s_list, out_len, out);
    }
}

// ============================================================================
// MULTI-VOICE MODE
// ============================================================================

static void alien_bang(t_alien *x) {
    if (x->x_num_voices == 0) return;
    
    // Find max pattern length
    int max_len = 1;
    for (int i = 0; i < x->x_num_voices; i++) {
        if (x->x_voices[i].length > max_len) max_len = x->x_voices[i].length;
    }
    
    // Output current value for each voice (in reverse order for Pd right-to-left)
    for (int i = x->x_num_voices - 1; i >= 0; i--) {
        t_voice *v = &x->x_voices[i];
        if (v->length > 0) {
            int idx = x->x_step % v->length;
            t_atom *current = &v->pattern[idx];
            if (current->a_type == A_FLOAT) {
                outlet_float(v->outlet, atom_getfloat(current));
            } else if (current->a_type == A_SYMBOL) {
                outlet_symbol(v->outlet, atom_getsymbol(current));
            }
        }
    }
    
    // Advance step
    x->x_step++;
    if (x->x_step >= max_len) {
        x->x_step = 0;
        outlet_bang(x->x_loop_out);
    }
}

static void alien_reset(t_alien *x) {
    x->x_step = 0;
}

// Apply buffered patterns (called after sync delay)
static void alien_sync_tick(t_alien *x) {
    for (int i = 0; i < x->x_num_voices; i++) {
        t_voice *v = &x->x_voices[i];
        if (v->has_buffer && v->buffer[0] != '\0') {
            v->length = alien_evaluate_to_atoms((t_alien*)x, v->buffer, v->pattern, ALIEN_MAX_STEPS);
            v->has_buffer = 0;
            v->buffer[0] = '\0';
        }
    }
    x->x_step = 0;  // Reset to start on sync
}

// Receive pattern for a voice via global send
static void alien_list(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (x->x_num_voices == 0) {
        // Generator mode: treat as pattern input
        if (argc > 0 && argv[0].a_type == A_SYMBOL) {
            alien_anything_generator(x, atom_getsymbol(&argv[0]), argc - 1, &argv[1]);
        }
        return;
    }
    
    // Multi-voice mode: this shouldn't happen directly, patterns come via voice names
}

// Called when a voice receives a pattern via [s name]
static void alien_voice_pattern(t_alien *x, t_symbol *voice_name, int argc, t_atom *argv) {
    // Find the voice
    t_voice *v = NULL;
    for (int i = 0; i < x->x_num_voices; i++) {
        if (x->x_voices[i].name == voice_name) {
            v = &x->x_voices[i];
            break;
        }
    }
    if (!v) return;
    
    // Build pattern string
    char *p = v->buffer;
    size_t remaining = sizeof(v->buffer) - 1;
    
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
    v->has_buffer = 1;
    
    // Schedule sync
    clock_delay(x->x_sync_clock, ALIEN_SYNC_DELAY_MS);
}

// Anything handler - routes based on mode
static void alien_anything(t_alien *x, t_symbol *s, int argc, t_atom *argv) {
    if (x->x_num_voices == 0) {
        // Generator mode
        alien_anything_generator(x, s, argc, argv);
    } else {
        // Multi-voice mode: check if this is a voice pattern
        alien_voice_pattern(x, s, argc, argv);
    }
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_alien *x = (t_alien *)pd_new(alien_class);
    
    x->x_num_voices = 0;
    x->x_step = 0;
    x->x_list_out = NULL;
    x->x_loop_out = NULL;
    x->x_sync_clock = clock_new(x, (t_method)alien_sync_tick);
    
    if (argc == 0) {
        // Generator mode: single list outlet
        x->x_list_out = outlet_new(&x->x_obj, &s_list);
    } else {
        // Multi-voice mode: one outlet per voice + loop outlet
        x->x_num_voices = (argc > ALIEN_MAX_VOICES) ? ALIEN_MAX_VOICES : argc;
        
        for (int i = 0; i < x->x_num_voices; i++) {
            t_voice *v = &x->x_voices[i];
            
            if (argv[i].a_type == A_SYMBOL) {
                v->name = atom_getsymbol(&argv[i]);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "v%d", i);
                v->name = gensym(buf);
            }
            
            v->outlet = outlet_new(&x->x_obj, &s_anything);
            v->length = 0;
            v->has_buffer = 0;
            v->buffer[0] = '\0';
            
            // Bind to voice name for global receive
            pd_bind(&x->x_obj.ob_pd, v->name);
        }
        
        // Loop outlet (last)
        x->x_loop_out = outlet_new(&x->x_obj, &s_bang);
        
        post("alien: %d voices", x->x_num_voices);
    }
    
    return (void *)x;
}

static void alien_free(t_alien *x) {
    clock_free(x->x_sync_clock);
    for (int i = 0; i < x->x_num_voices; i++) {
        pd_unbind(&x->x_obj.ob_pd, x->x_voices[i].name);
    }
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
    class_addlist(alien_class, alien_list);
    class_addanything(alien_class, alien_anything);
    class_addmethod(alien_class, (t_method)alien_reset, gensym("reset"), 0);
    
    post("alien %s - polymorphic pattern sequencer", ALIEN_VERSION_STRING);
}
