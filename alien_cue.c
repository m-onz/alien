/*
 * alien_cue - globally registered cue bundle for Pure Data
 *
 *   [alien_cue 4]
 *
 * Registers itself in a global registry in creation order; its position
 * (1-based) is its cue index. When launched by alien_orchestrate or a
 * bang, bangs all outlets right-to-left.
 *
 * Creation arg: number of outlets (1-64, default 1)
 * bang:         launch this cue
 * index:        post this cue's registry position
 */

#include "alien_orch.h"

static t_class *alien_cue_class;

static void cue_bang(t_alien_cue *x) {
    alien_orch_launch_cue(x);
}

static void cue_index(t_alien_cue *x) {
    int idx = alien_orch_cue_index(x);
    if (idx > 0)
        post("alien_cue: index %d of %d", idx, alien_orch_registry()->r_count);
    else
        pd_error(x, "alien_cue: not registered");
}

static void *cue_new(t_floatarg f) {
    t_alien_cue *x = (t_alien_cue *)pd_new(alien_cue_class);
    int n = (int)f;
    if (n < 1) n = 1;
    if (n > ALIEN_CUE_MAX_OUTLETS) {
        pd_error(x, "alien_cue: maximum %d outlets", ALIEN_CUE_MAX_OUTLETS);
        n = ALIEN_CUE_MAX_OUTLETS;
    }
    x->x_noutlets = n;
    for (int i = 0; i < n; i++)
        x->x_out[i] = outlet_new(&x->x_obj, &s_bang);
    alien_orch_register_cue(x);
    return (void *)x;
}

static void cue_free(t_alien_cue *x) {
    alien_orch_unregister_cue(x);
}

void alien_cue_setup(void) {
    alien_cue_class = class_new(gensym("alien_cue"),
        (t_newmethod)cue_new,
        (t_method)cue_free,
        sizeof(t_alien_cue),
        CLASS_DEFAULT,
        A_DEFFLOAT,
        0);

    class_addbang(alien_cue_class, cue_bang);
    class_addmethod(alien_cue_class, (t_method)cue_index, gensym("index"), 0);
}
