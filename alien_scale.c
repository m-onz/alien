/*
 * alien_scale.c - Scale/mode quantizer for Pure Data
 *
 * Maps incoming integers (scale degrees) to pitches in current scale.
 *
 * Inlets:
 *   1: float/int - scale degree to map
 *   2: messages - scale/mode/root configuration
 *
 * Outlets:
 *   1: float - mapped MIDI pitch
 *
 * Messages:
 *   root <n>           - set root note (default 60 = middle C)
 *   scale <name>       - set scale by name (major, minor, dorian, etc.)
 *   scale <n n n ...>  - set scale by intervals from root
 *   mode <name>        - alias for scale
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
// SCALE DEGREE TO PITCH CONVERSION
// ============================================================================

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

// ============================================================================
// MESSAGE HANDLERS
// ============================================================================

static void alien_scale_float(t_alien_scale *x, t_floatarg f) {
    int degree = (int)f;
    int pitch = degree_to_pitch(x, degree);
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

// Alias for scale
static void alien_scale_mode(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    alien_scale_scale(x, s, argc, argv);
}

// Handle lists (from alien output)
static void alien_scale_list(t_alien_scale *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused parameter
    // Output a list of mapped pitches
    t_atom *out = (t_atom *)getbytes(sizeof(t_atom) * argc);

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int degree = (int)atom_getfloat(&argv[i]);
            SETFLOAT(&out[i], degree_to_pitch(x, degree));
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

    // Defaults: C major
    x->x_root = 60;
    x->x_scale_len = 7;
    int major[] = {0, 2, 4, 5, 7, 9, 11};
    for (int i = 0; i < 7; i++) x->x_scale[i] = major[i];

    // Parse creation arguments: [root] [scale]
    int arg_idx = 0;

    if (argc > arg_idx && argv[arg_idx].a_type == A_FLOAT) {
        x->x_root = (int)atom_getfloat(&argv[arg_idx]);
        arg_idx++;
    }

    if (argc > arg_idx) {
        // Remaining args are scale
        alien_scale_scale(x, gensym("scale"), argc - arg_idx, argv + arg_idx);
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
    post("alien_scale %s - scale/mode quantizer", ALIEN_VERSION_STRING);
}
