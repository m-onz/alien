/*
 * alien_orch.h - shared cue registry for alien_cue / alien_orchestrate
 *
 * The two externals are separate binaries, so they cannot share C globals.
 * The registry lives in a hidden Pd object bound to a reserved symbol:
 * whichever external touches it first creates it, and later lookups find
 * the same instance through Pd's symbol table.
 */

#ifndef ALIEN_ORCH_H
#define ALIEN_ORCH_H

#include "m_pd.h"

#define ALIEN_CUE_MAX_OUTLETS 64
#define ALIEN_ORCH_REGISTRY_SYM "__alien_cue_registry"
#define ALIEN_ORCH_REGISTRY_INIT_CAP 16

typedef struct _alien_cue {
    t_object x_obj;
    int x_noutlets;
    t_outlet *x_out[ALIEN_CUE_MAX_OUTLETS];
} t_alien_cue;

typedef struct _alien_cue_registry {
    t_pd r_pd;              /* bindable header — must stay first */
    t_alien_cue **r_cues;   /* creation order; position defines cue index */
    int r_count;
    int r_capacity;
} t_alien_cue_registry;

/* per-binary copy of the (method-less) registry class */
static t_class *alien_orch_registry_class = NULL;

static t_alien_cue_registry *alien_orch_registry(void) {
    t_symbol *s = gensym(ALIEN_ORCH_REGISTRY_SYM);
    if (s->s_thing)
        return (t_alien_cue_registry *)s->s_thing;
    if (!alien_orch_registry_class) {
        alien_orch_registry_class = class_new(gensym("_alien_cue_registry"),
            0, 0, sizeof(t_alien_cue_registry), CLASS_PD, 0);
    }
    t_alien_cue_registry *r =
        (t_alien_cue_registry *)pd_new(alien_orch_registry_class);
    r->r_capacity = ALIEN_ORCH_REGISTRY_INIT_CAP;
    r->r_count = 0;
    r->r_cues = (t_alien_cue **)getbytes(
        sizeof(t_alien_cue *) * r->r_capacity);
    pd_bind(&r->r_pd, s);
    return r;
}

static void alien_orch_register_cue(t_alien_cue *x) {
    t_alien_cue_registry *r = alien_orch_registry();
    if (r->r_count >= r->r_capacity) {
        int old_cap = r->r_capacity;
        r->r_capacity *= 2;
        r->r_cues = (t_alien_cue **)resizebytes(r->r_cues,
            sizeof(t_alien_cue *) * old_cap,
            sizeof(t_alien_cue *) * r->r_capacity);
    }
    r->r_cues[r->r_count++] = x;
}

/* remove and compact — order must be preserved, indices depend on it */
static void alien_orch_unregister_cue(t_alien_cue *x) {
    t_alien_cue_registry *r = alien_orch_registry();
    for (int i = 0; i < r->r_count; i++) {
        if (r->r_cues[i] == x) {
            for (int j = i; j < r->r_count - 1; j++)
                r->r_cues[j] = r->r_cues[j + 1];
            r->r_count--;
            return;
        }
    }
}

/* 1-based position in the registry, 0 if not found */
static int alien_orch_cue_index(t_alien_cue *x) {
    t_alien_cue_registry *r = alien_orch_registry();
    for (int i = 0; i < r->r_count; i++)
        if (r->r_cues[i] == x) return i + 1;
    return 0;
}

/* bang all outlets right-to-left (Pd trigger convention) */
static void alien_orch_launch_cue(t_alien_cue *x) {
    for (int i = x->x_noutlets - 1; i >= 0; i--)
        outlet_bang(x->x_out[i]);
}

#endif /* ALIEN_ORCH_H */
