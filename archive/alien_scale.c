/*
 * alien_scale.c - Scale/mode quantizer for Pure Data
 *
 * Maps incoming values to pitches in current scale.
 *
 * Modes:
 *   degree   - input is scale degree index (default)
 *   quantize - input is MIDI note, snapped to nearest scale pitch
 *
 * Inlets:
 *   1: float/int - scale degree (degree mode) or MIDI note (quantize mode)
 *   2: messages - configuration
 *
 * Outlets:
 *   1: float - mapped MIDI pitch
 *
 * Messages:
 *   root <n>            - set root note (default 60 = middle C)
 *   scale <name>        - set scale by name (major, minor, dorian, etc.)
 *   scale <n n n ...>   - set scale by intervals from root
 *   mode degree         - input = scale degree index (default)
 *   mode quantize       - input = MIDI note, nearest in-scale pitch
 *   mode <scale_name>   - alias for scale (backward compat)
 *   degrees <n n ...>   - set active degrees (harmonic field filter)
 *   degrees off         - disable degree filter (all degrees active)
 *   ref <n>             - set reference pitch for interval filter
 *   intervals <n ...>   - set allowed intervals from reference (0-11)
 *   intervals off       - disable interval filter
 *   range <lo> <hi>     - set output MIDI range (default 40 79)
 */

#include "m_pd.h"
#include "alien_core.h"
#include <string.h>

static t_class *alien_scale_class;

#define MAX_SCALE_LEN 12

typedef struct _alien_scale {
    t_object x_obj;
    t_outlet *x_out;

    int x_root;                      // Root note (MIDI pitch)
    int x_scale[MAX_SCALE_LEN];      // Intervals from root
    int x_scale_len;                 // Number of notes in scale

    int x_mode;                      // 0 = degree, 1 = quantize

    int x_degrees[MAX_SCALE_LEN];    // Which degrees are active
    int x_degree_count;              // Number of active degrees
    int x_degree_filter_on;          // 0 = off, 1 = on

    int x_reference;                 // Reference MIDI pitch (default: root)
    int x_intervals[12];             // Allowed intervals 0-11 (pitch classes)
    int x_interval_count;
    int x_interval_filter_on;        // 0 = off, 1 = on

    int x_range_lo;                  // Output range low bound (MIDI)
    int x_range_hi;                  // Output range high bound (MIDI)
} t_alien_scale;

// ============================================================================
// BUILT-IN SCALES
// ============================================================================

typedef struct {
    const char *name;
    int intervals[MAX_SCALE_LEN];
    int len;
} scale_def;

static const scale_def scales[] = {
    // Major modes
    {"major",       {0, 2, 4, 5, 7, 9, 11}, 7},
    {"ionian",      {0, 2, 4, 5, 7, 9, 11}, 7},
    {"dorian",      {0, 2, 3, 5, 7, 9, 10}, 7},
    {"phrygian",    {0, 1, 3, 5, 7, 8, 10}, 7},
    {"lydian",      {0, 2, 4, 6, 7, 9, 11}, 7},
    {"mixolydian",  {0, 2, 4, 5, 7, 9, 10}, 7},
    {"aeolian",     {0, 2, 3, 5, 7, 8, 10}, 7},
    {"minor",       {0, 2, 3, 5, 7, 8, 10}, 7},
    {"locrian",     {0, 1, 3, 5, 6, 8, 10}, 7},

    // Pentatonic
    {"pentatonic",  {0, 2, 4, 7, 9}, 5},
    {"pent",        {0, 2, 4, 7, 9}, 5},
    {"minpent",     {0, 3, 5, 7, 10}, 5},

    // Other common scales
    {"blues",       {0, 3, 5, 6, 7, 10}, 6},
    {"chromatic",   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 12},
    {"wholetone",   {0, 2, 4, 6, 8, 10}, 6},
    {"diminished",  {0, 2, 3, 5, 6, 8, 9, 11}, 8},
    {"augmented",   {0, 3, 4, 7, 8, 11}, 6},

    // Harmonic/melodic minor
    {"harmonic",    {0, 2, 3, 5, 7, 8, 11}, 7},
    {"melodic",     {0, 2, 3, 5, 7, 9, 11}, 7},

    // World scales
    {"hirajoshi",   {0, 2, 3, 7, 8}, 5},
    {"insen",       {0, 1, 5, 7, 10}, 5},
    {"iwato",       {0, 1, 5, 6, 10}, 5},
    {"hungarian",   {0, 2, 3, 6, 7, 8, 11}, 7},
    {"romanian",    {0, 2, 3, 6, 7, 9, 10}, 7},
    {"phrygdom",    {0, 1, 4, 5, 7, 8, 10}, 7},  // Phrygian dominant

    {NULL, {0}, 0}  // Terminator
};

// ============================================================================
// PITCH CONVERSION HELPERS
// ============================================================================

// Find nearest pitch class from a set of pitch classes.
// Returns the signed semitone adjustment (typically -6 to +5).
// If out_index is non-NULL, stores the index of the best match.
static int nearest_pc(int pc, const int *targets, int ntargets, int *out_index) {
    int best_adj = 0;
    int best_dist = 13;
    int best_idx = 0;
    for (int i = 0; i < ntargets; i++) {
        int fwd = (targets[i] - pc + 12) % 12;
        int bwd = (pc - targets[i] + 12) % 12;
        int dist, adj;
        if (fwd <= bwd) { dist = fwd; adj = fwd; }
        else { dist = bwd; adj = -bwd; }
        if (dist < best_dist) {
            best_dist = dist;
            best_adj = adj;
            best_idx = i;
        }
    }
    if (out_index) *out_index = best_idx;
    return best_adj;
}

// Convert scale degree index to MIDI pitch (degree mode core)
static int degree_to_pitch(t_alien_scale *x, int degree) {
    if (x->x_scale_len == 0) return x->x_root + degree;  // Chromatic fallback

    // Handle negative degrees and octave wrapping
    int octave = 0;
    int deg = degree;

    if (deg >= 0) {
        octave = deg / x->x_scale_len;
        deg = deg % x->x_scale_len;
    } else {
        // For negative: -1 should be highest note of octave below
        // E.g., in 7-note scale: -1 -> degree 6, octave -1
        octave = (deg + 1) / x->x_scale_len - 1;
        deg = deg % x->x_scale_len;
        if (deg < 0) deg += x->x_scale_len;
    }

    return x->x_root + (octave * 12) + x->x_scale[deg];
}

// Quantize MIDI note to nearest scale pitch.
// Returns the adjusted MIDI note and (optionally) the scale degree index.
static int quantize_midi_to_scale(t_alien_scale *x, int midi, int *out_degree) {
    if (x->x_scale_len == 0) {
        if (out_degree) *out_degree = 0;
        return midi;
    }

    int pc = ((midi - x->x_root) % 12 + 12) % 12;
    int deg_idx;
    int adj = nearest_pc(pc, x->x_scale, x->x_scale_len, &deg_idx);

    if (out_degree) *out_degree = deg_idx;
    return midi + adj;
}

// ============================================================================
// PROCESSING PIPELINE
// ============================================================================

static int process_input(t_alien_scale *x, int input) {
    int pitch;

    if (x->x_mode == 1) {
        // === QUANTIZE MODE ===
        // Step 1: Snap MIDI note to nearest scale tone
        int degree_idx;
        pitch = quantize_midi_to_scale(x, input, &degree_idx);

        // Step 2: Degree filter
        if (x->x_degree_filter_on && x->x_degree_count > 0) {
            int is_active = 0;
            for (int i = 0; i < x->x_degree_count; i++) {
                if (x->x_degrees[i] == degree_idx) {
                    is_active = 1;
                    break;
                }
            }
            if (!is_active) {
                // Build array of active degree pitch classes
                int active_pcs[MAX_SCALE_LEN];
                int nactive = 0;
                for (int i = 0; i < x->x_degree_count; i++) {
                    int deg = x->x_degrees[i];
                    if (deg >= 0 && deg < x->x_scale_len) {
                        active_pcs[nactive++] = x->x_scale[deg];
                    }
                }
                if (nactive > 0) {
                    int pc = ((pitch - x->x_root) % 12 + 12) % 12;
                    int adj = nearest_pc(pc, active_pcs, nactive, NULL);
                    pitch += adj;
                }
            }
        }
    } else {
        // === DEGREE MODE ===
        int degree = input;

        // Step 1: Degree filter (snap to nearest active degree)
        if (x->x_degree_filter_on && x->x_degree_count > 0
            && x->x_scale_len > 0) {
            int octave, norm_deg;
            if (degree >= 0) {
                octave = degree / x->x_scale_len;
                norm_deg = degree % x->x_scale_len;
            } else {
                octave = (degree + 1) / x->x_scale_len - 1;
                norm_deg = degree % x->x_scale_len;
                if (norm_deg < 0) norm_deg += x->x_scale_len;
            }

            int is_active = 0;
            for (int i = 0; i < x->x_degree_count; i++) {
                if (x->x_degrees[i] == norm_deg) {
                    is_active = 1;
                    break;
                }
            }

            if (!is_active) {
                int best = x->x_degrees[0];
                int best_dist = x->x_scale_len;
                for (int i = 0; i < x->x_degree_count; i++) {
                    int d = x->x_degrees[i];
                    if (d >= 0 && d < x->x_scale_len) {
                        int diff = abs(norm_deg - d);
                        int wrap = x->x_scale_len - diff;
                        int dist = (diff < wrap) ? diff : wrap;
                        if (dist < best_dist) {
                            best_dist = dist;
                            best = d;
                        }
                    }
                }
                degree = octave * x->x_scale_len + best;
            }
        }

        // Step 2: Convert degree to pitch
        pitch = degree_to_pitch(x, degree);
    }

    // Step 3: Interval filter (both modes)
    if (x->x_interval_filter_on && x->x_interval_count > 0) {
        int interval_from_ref = ((pitch - x->x_reference) % 12 + 12) % 12;

        int is_allowed = 0;
        for (int i = 0; i < x->x_interval_count; i++) {
            if (x->x_intervals[i] == interval_from_ref) {
                is_allowed = 1;
                break;
            }
        }

        if (!is_allowed) {
            int adj = nearest_pc(interval_from_ref, x->x_intervals,
                                 x->x_interval_count, NULL);
            pitch += adj;
        }
    }

    // Step 4: Octave-fold into output range
    while (pitch > x->x_range_hi) pitch -= 12;
    while (pitch < x->x_range_lo) pitch += 12;

    return pitch;
}

// ============================================================================
// MESSAGE HANDLERS
// ============================================================================

static void alien_scale_float(t_alien_scale *x, t_floatarg f) {
    int input = (int)f;
    int pitch = process_input(x, input);
    outlet_float(x->x_out, pitch);
}

static void alien_scale_root(t_alien_scale *x, t_floatarg f) {
    x->x_root = (int)f;
}

static void alien_scale_scale(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused parameter
    if (argc == 0) return;

    // Check if first arg is a symbol (scale name) or number (intervals)
    if (argv[0].a_type == A_SYMBOL) {
        const char *name = atom_getsymbol(&argv[0])->s_name;

        // Search for scale by name
        for (int i = 0; scales[i].name != NULL; i++) {
            if (strcmp(scales[i].name, name) == 0) {
                x->x_scale_len = scales[i].len;
                for (int j = 0; j < scales[i].len; j++) {
                    x->x_scale[j] = scales[i].intervals[j];
                }
                return;
            }
        }
        pd_error(x, "alien_scale: unknown scale '%s'", name);
    } else {
        // Numeric intervals provided
        x->x_scale_len = (argc > MAX_SCALE_LEN) ? MAX_SCALE_LEN : argc;
        for (int i = 0; i < x->x_scale_len; i++) {
            x->x_scale[i] = (int)atom_getfloat(&argv[i]);
        }
    }
}

// Set input mode or fall through to scale (backward compat)
static void alien_scale_mode(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc >= 1 && argv[0].a_type == A_SYMBOL) {
        const char *name = atom_getsymbol(&argv[0])->s_name;
        if (strcmp(name, "degree") == 0) {
            x->x_mode = 0;
            return;
        }
        if (strcmp(name, "quantize") == 0) {
            x->x_mode = 1;
            return;
        }
    }
    // Fall through: treat as scale name for backward compatibility
    alien_scale_scale(x, s, argc, argv);
}

// Set active degrees for harmonic field filtering
static void alien_scale_degrees(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc == 0) return;

    // Check for "off"
    if (argc == 1 && argv[0].a_type == A_SYMBOL) {
        if (strcmp(atom_getsymbol(&argv[0])->s_name, "off") == 0) {
            x->x_degree_filter_on = 0;
            return;
        }
    }

    // Set active degrees
    x->x_degree_count = 0;
    for (int i = 0; i < argc && x->x_degree_count < MAX_SCALE_LEN; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int deg = (int)atom_getfloat(&argv[i]);
            if (deg >= 0 && deg < x->x_scale_len) {
                x->x_degrees[x->x_degree_count++] = deg;
            }
        }
    }
    x->x_degree_filter_on = (x->x_degree_count > 0) ? 1 : 0;
}

// Set reference pitch for interval filter
static void alien_scale_ref(t_alien_scale *x, t_floatarg f) {
    x->x_reference = (int)f;
}

// Set output pitch range (octave-folding bounds)
static void alien_scale_range(t_alien_scale *x, t_floatarg lo, t_floatarg hi) {
    x->x_range_lo = (int)lo;
    x->x_range_hi = (int)hi;
    if (x->x_range_hi - x->x_range_lo < 12) {
        pd_error(x, "alien_scale: range must span at least 12 semitones");
        x->x_range_lo = 40;
        x->x_range_hi = 79;
    }
}

// Set allowed intervals from reference pitch
static void alien_scale_intervals(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc == 0) return;

    // Check for "off"
    if (argc == 1 && argv[0].a_type == A_SYMBOL) {
        if (strcmp(atom_getsymbol(&argv[0])->s_name, "off") == 0) {
            x->x_interval_filter_on = 0;
            return;
        }
    }

    x->x_interval_count = 0;
    for (int i = 0; i < argc && x->x_interval_count < 12; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int interval = (int)atom_getfloat(&argv[i]);
            if (interval >= 0 && interval < 12) {
                x->x_intervals[x->x_interval_count++] = interval;
            }
        }
    }
    x->x_interval_filter_on = (x->x_interval_count > 0) ? 1 : 0;
}

// Handle lists (from alien output)
static void alien_scale_list(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused parameter
    // Output a list of mapped pitches
    t_atom *out = (t_atom *)getbytes(sizeof(t_atom) * argc);

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int input = (int)atom_getfloat(&argv[i]);
            SETFLOAT(&out[i], process_input(x, input));
        } else {
            // Pass through symbols (like "-" for rests)
            out[i] = argv[i];
        }
    }

    outlet_list(x->x_out, &s_list, argc, out);
    freebytes(out, sizeof(t_atom) * argc);
}

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

static void *alien_scale_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused parameter
    t_alien_scale *x = (t_alien_scale *)pd_new(alien_scale_class);

    // Defaults: C major, degree mode
    x->x_root = 60;
    x->x_scale_len = 7;
    int major[] = {0, 2, 4, 5, 7, 9, 11};
    for (int i = 0; i < 7; i++) x->x_scale[i] = major[i];

    x->x_mode = 0;
    x->x_degree_count = 0;
    x->x_degree_filter_on = 0;
    x->x_reference = 60;
    x->x_interval_count = 0;
    x->x_interval_filter_on = 0;
    x->x_range_lo = 40;
    x->x_range_hi = 79;

    // Check for "quantize" as last creation argument
    int effective_argc = argc;
    if (argc > 0 && argv[argc - 1].a_type == A_SYMBOL) {
        if (strcmp(atom_getsymbol(&argv[argc - 1])->s_name, "quantize") == 0) {
            x->x_mode = 1;
            effective_argc--;
        }
    }

    // Parse creation arguments: [root] [scale_name_or_intervals...]
    int arg_idx = 0;

    if (effective_argc > arg_idx && argv[arg_idx].a_type == A_FLOAT) {
        x->x_root = (int)atom_getfloat(&argv[arg_idx]);
        x->x_reference = x->x_root;
        arg_idx++;
    }

    if (effective_argc > arg_idx) {
        // Remaining args are scale
        alien_scale_scale(x, gensym("scale"), effective_argc - arg_idx, argv + arg_idx);
    }

    x->x_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

// ============================================================================
// SETUP
// ============================================================================

void alien_scale_setup(void) {
    alien_scale_class = class_new(
        gensym("alien_scale"),
        (t_newmethod)alien_scale_new,
        NULL,
        sizeof(t_alien_scale),
        CLASS_DEFAULT,
        A_GIMME, 0
    );

    class_addfloat(alien_scale_class, alien_scale_float);
    class_addlist(alien_scale_class, alien_scale_list);
    class_addmethod(alien_scale_class, (t_method)alien_scale_root,
                    gensym("root"), A_FLOAT, 0);
    class_addmethod(alien_scale_class, (t_method)alien_scale_scale,
                    gensym("scale"), A_GIMME, 0);
    class_addmethod(alien_scale_class, (t_method)alien_scale_mode,
                    gensym("mode"), A_GIMME, 0);
    class_addmethod(alien_scale_class, (t_method)alien_scale_degrees,
                    gensym("degrees"), A_GIMME, 0);
    class_addmethod(alien_scale_class, (t_method)alien_scale_ref,
                    gensym("ref"), A_FLOAT, 0);
    class_addmethod(alien_scale_class, (t_method)alien_scale_intervals,
                    gensym("intervals"), A_GIMME, 0);
    class_addmethod(alien_scale_class, (t_method)alien_scale_range,
                    gensym("range"), A_FLOAT, A_FLOAT, 0);
    post("alien_scale %s - scale/mode quantizer", ALIEN_VERSION_STRING);
}
