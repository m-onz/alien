/*
 * alien_orchestrate - launch alien_cue bundles by wrapped numeric index
 *
 *   [alien_orchestrate]
 *
 * Receives cue indices (1-based), wraps them over the registered cue
 * count, and launches the selected cue. Feed it from a number box or
 * from an alien pattern via [else/sequencer] so the DSL can drive
 * high-level composition changes:
 *
 *   (seq 1 - - - 2 - - - 3 3 4 -)
 *
 * float n:  wrap n over cue count, launch that cue (every number retriggers)
 * - . _:    rest, do nothing
 * bang:     relaunch last selected cue
 * reset:    clear last selected cue
 * count:    post registered cue count
 * dump:     post cue registry in order
 * Outlet:   launched cue index (1-based, after wrapping)
 */

#include "alien_orch.h"
#include <string.h>

static t_class *alien_orchestrate_class;

typedef struct _alien_orchestrate {
    t_object x_obj;
    t_outlet *x_index_out;
    int x_last;         /* last raw input, re-wrapped on bang */
    int x_has_last;
    int x_warned;       /* suppress repeated no-cues errors */
} t_alien_orchestrate;

static int orch_is_rest(const char *s) {
    return (s[1] == '\0' && (s[0] == '-' || s[0] == '.' || s[0] == '_'));
}

static void orch_launch(t_alien_orchestrate *x, int input) {
    t_alien_cue_registry *r = alien_orch_registry();
    if (r->r_count <= 0) {
        if (!x->x_warned) {
            pd_error(x, "alien_orchestrate: no cues registered");
            x->x_warned = 1;
        }
        return;
    }
    x->x_warned = 0;
    int wrapped = (input - 1) % r->r_count;
    if (wrapped < 0) wrapped += r->r_count;
    x->x_last = input;
    x->x_has_last = 1;
    outlet_float(x->x_index_out, (t_float)(wrapped + 1));
    alien_orch_launch_cue(r->r_cues[wrapped]);
}

static void orch_float(t_alien_orchestrate *x, t_floatarg f) {
    orch_launch(x, (int)f);
}

static void orch_bang(t_alien_orchestrate *x) {
    if (x->x_has_last) orch_launch(x, x->x_last);
}

static void orch_symbol(t_alien_orchestrate *x, t_symbol *s) {
    if (orch_is_rest(s->s_name)) return;
    pd_error(x, "alien_orchestrate: unknown symbol '%s'", s->s_name);
}

static void orch_list(t_alien_orchestrate *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc < 1) return;
    if (argv[0].a_type == A_FLOAT)
        orch_launch(x, (int)atom_getfloat(&argv[0]));
    else if (argv[0].a_type == A_SYMBOL)
        orch_symbol(x, atom_getsymbol(&argv[0]));
}

static void orch_anything(t_alien_orchestrate *x, t_symbol *s, int argc, t_atom *argv) {
    (void)argc; (void)argv;
    orch_symbol(x, s);
}

static void orch_reset(t_alien_orchestrate *x) {
    x->x_has_last = 0;
}

static void orch_count(t_alien_orchestrate *x) {
    (void)x;
    post("alien_orchestrate: %d cue%s registered",
        alien_orch_registry()->r_count,
        alien_orch_registry()->r_count == 1 ? "" : "s");
}

static void orch_dump(t_alien_orchestrate *x) {
    (void)x;
    t_alien_cue_registry *r = alien_orch_registry();
    if (r->r_count == 0) {
        post("alien_orchestrate: no cues registered");
        return;
    }
    for (int i = 0; i < r->r_count; i++)
        post("alien_orchestrate: cue %d -> [alien_cue %d]",
            i + 1, r->r_cues[i]->x_noutlets);
}

static void *orch_new(void) {
    t_alien_orchestrate *x = (t_alien_orchestrate *)pd_new(alien_orchestrate_class);
    x->x_last = 0;
    x->x_has_last = 0;
    x->x_warned = 0;
    x->x_index_out = outlet_new(&x->x_obj, &s_float);
    return (void *)x;
}

void alien_orchestrate_setup(void) {
    alien_orchestrate_class = class_new(gensym("alien_orchestrate"),
        (t_newmethod)orch_new,
        0,
        sizeof(t_alien_orchestrate),
        CLASS_DEFAULT,
        0);

    class_addbang(alien_orchestrate_class, orch_bang);
    class_addfloat(alien_orchestrate_class, orch_float);
    class_addsymbol(alien_orchestrate_class, orch_symbol);
    class_addlist(alien_orchestrate_class, orch_list);
    class_addanything(alien_orchestrate_class, orch_anything);
    class_addmethod(alien_orchestrate_class, (t_method)orch_reset, gensym("reset"), 0);
    class_addmethod(alien_orchestrate_class, (t_method)orch_count, gensym("count"), 0);
    class_addmethod(alien_orchestrate_class, (t_method)orch_dump, gensym("dump"), 0);
}
