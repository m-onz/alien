/*
 * alien_cluster.c - Cluster chord generator for Pure Data
 *
 * Generates diatonic cluster chords with compatible bass notes.
 * Modulates harmonically based on previous chord.
 *
 * Inlets:
 *   1: bang - generate new cluster chord
 *      float - set root and generate
 *
 * Outlets:
 *   1 (left): list - cluster chord (MIDI notes)
 *   2 (right): list - compatible bass notes
 *
 * Creation: [alien_cluster] or [alien_cluster 48] (root note)
 */

#include "m_pd.h"
#include <stdlib.h>
#include <time.h>

#define ALIEN_VERSION_STRING "0.2.1"

static t_class *alien_cluster_class;

#define CLUSTER_SIZE 4

typedef struct _alien_cluster {
    t_object x_obj;
    t_outlet *x_chord_out;
    t_outlet *x_bass_out;

    int x_root;
    int x_last_root;
    int x_random_initialized;
} t_alien_cluster;

// Major scale intervals
static const int major_scale[] = {0, 2, 4, 5, 7, 9, 11};

// Pleasing modulation intervals (fifths, fourths, relative minor/major)
static const int mod_intervals[] = {0, 7, 5, 4, 9, 2};
static const int num_mod_intervals = 6;

// ============================================================================
// RANDOM
// ============================================================================

static void ensure_random_init(t_alien_cluster *x) {
    if (!x->x_random_initialized) {
        srand((unsigned int)time(NULL) ^ (unsigned int)(intptr_t)x);
        x->x_random_initialized = 1;
    }
}

static int random_range(t_alien_cluster *x, int min, int max) {
    ensure_random_init(x);
    if (max <= min) return min;
    return min + (rand() % (max - min + 1));
}

// ============================================================================
// CHORD GENERATION
// ============================================================================

static int scale_degree_to_pitch(int root, int degree) {
    int octave = degree / 7;
    int deg = degree % 7;
    if (deg < 0) { deg += 7; octave--; }
    return root + octave * 12 + major_scale[deg];
}

static void alien_cluster_bang(t_alien_cluster *x) {
    // Modulate root harmonically from last chord
    int mod_idx = random_range(x, 0, num_mod_intervals - 1);
    int new_root = (x->x_last_root + mod_intervals[mod_idx]) % 12;
    new_root += (x->x_root / 12) * 12;  // Keep in same octave range

    // Build cluster: 3-4 adjacent scale degrees, one octave up
    int cluster_base = new_root + 12;
    int start_degree = random_range(x, 0, 4);  // Start on degrees 0-4

    int chord[CLUSTER_SIZE];
    for (int i = 0; i < CLUSTER_SIZE; i++) {
        chord[i] = scale_degree_to_pitch(cluster_base, start_degree + i);
    }

    // Generate bass notes (root, fifth, fourth)
    int bass_root = new_root - 12;
    if (bass_root < 36) bass_root += 12;

    t_atom bass[3];
    SETFLOAT(&bass[0], bass_root);
    SETFLOAT(&bass[1], bass_root + 7);  // Fifth
    SETFLOAT(&bass[2], bass_root + 5);  // Fourth

    // Store for next modulation
    x->x_last_root = new_root;

    // Output (right first per Pd convention)
    outlet_list(x->x_bass_out, &s_list, 3, bass);

    t_atom out[CLUSTER_SIZE];
    for (int i = 0; i < CLUSTER_SIZE; i++) {
        SETFLOAT(&out[i], chord[i]);
    }
    outlet_list(x->x_chord_out, &s_list, CLUSTER_SIZE, out);
}

static void alien_cluster_float(t_alien_cluster *x, t_floatarg f) {
    x->x_root = (int)f;
    x->x_last_root = x->x_root;
    alien_cluster_bang(x);
}

// ============================================================================
// CONSTRUCTOR
// ============================================================================

static void *alien_cluster_new(t_floatarg f) {
    t_alien_cluster *x = (t_alien_cluster *)pd_new(alien_cluster_class);

    x->x_root = (f > 0) ? (int)f : 60;
    x->x_last_root = x->x_root;
    x->x_random_initialized = 0;

    x->x_chord_out = outlet_new(&x->x_obj, &s_list);
    x->x_bass_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

// ============================================================================
// SETUP
// ============================================================================

void alien_cluster_setup(void) {
    alien_cluster_class = class_new(
        gensym("alien_cluster"),
        (t_newmethod)alien_cluster_new,
        NULL,
        sizeof(t_alien_cluster),
        CLASS_DEFAULT,
        A_DEFFLOAT, 0
    );

    class_addbang(alien_cluster_class, alien_cluster_bang);
    class_addfloat(alien_cluster_class, alien_cluster_float);

    post("alien_cluster %s - cluster chord generator", ALIEN_VERSION_STRING);
}
