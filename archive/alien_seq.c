/*
 * alien_seq - Simple step sequencer for Pure Data
 *
 * Replaces else/sequencer dependency for the alien framework.
 *
 * Usage:
 *   [alien_seq]      - default 16 steps
 *   [alien_seq 8]    - 8 step sequencer
 *
 * Inlets:
 *   1. bang - advance to next step, output current value
 *      list - set pattern (list of numbers/symbols)
 *      reset - go back to step 0
 *   2. float - set step position directly
 *
 * Outlets:
 *   1. current value (float or symbol "-" for rest)
 *   2. bang when sequence loops back to start
 *   3. current step number (0-indexed)
 */

#include "m_pd.h"
#include <string.h>

static t_class *alien_seq_class;

#define ALIEN_SEQ_MAX_STEPS 1024

typedef struct _alien_seq {
    t_object x_obj;
    t_outlet *x_val_out;      // current value
    t_outlet *x_loop_out;     // bang on loop
    t_outlet *x_step_out;     // current step number
    t_atom x_pattern[ALIEN_SEQ_MAX_STEPS];
    int x_length;             // pattern length
    int x_step;               // current step
    int x_max_steps;          // max steps (creation arg)
} t_alien_seq;

static void alien_seq_output_current(t_alien_seq *x) {
    if (x->x_length == 0) return;
    
    // Output step number
    outlet_float(x->x_step_out, x->x_step);
    
    // Output current value
    t_atom *current = &x->x_pattern[x->x_step];
    if (current->a_type == A_FLOAT) {
        outlet_float(x->x_val_out, atom_getfloat(current));
    } else if (current->a_type == A_SYMBOL) {
        outlet_symbol(x->x_val_out, atom_getsymbol(current));
    }
}

static void alien_seq_bang(t_alien_seq *x) {
    if (x->x_length == 0) return;
    
    // Output current value
    alien_seq_output_current(x);
    
    // Advance step
    x->x_step++;
    if (x->x_step >= x->x_length) {
        x->x_step = 0;
        outlet_bang(x->x_loop_out);
    }
}

static void alien_seq_list(t_alien_seq *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    
    if (argc > ALIEN_SEQ_MAX_STEPS) {
        pd_error(x, "alien_seq: pattern too long (max %d)", ALIEN_SEQ_MAX_STEPS);
        argc = ALIEN_SEQ_MAX_STEPS;
    }
    
    // Copy pattern
    for (int i = 0; i < argc; i++) {
        x->x_pattern[i] = argv[i];
    }
    x->x_length = argc;
    x->x_step = 0;
}

static void alien_seq_reset(t_alien_seq *x) {
    x->x_step = 0;
}

static void alien_seq_set_step(t_alien_seq *x, t_floatarg f) {
    int step = (int)f;
    if (x->x_length > 0) {
        step = step % x->x_length;
        if (step < 0) step += x->x_length;
        x->x_step = step;
    }
}

static void alien_seq_float(t_alien_seq *x, t_floatarg f) {
    // Single float sets a 1-element pattern
    SETFLOAT(&x->x_pattern[0], f);
    x->x_length = 1;
    x->x_step = 0;
}

static void alien_seq_symbol(t_alien_seq *x, t_symbol *s) {
    // Single symbol (like "-") sets a 1-element pattern
    SETSYMBOL(&x->x_pattern[0], s);
    x->x_length = 1;
    x->x_step = 0;
}

static void *alien_seq_new(t_floatarg max_steps) {
    t_alien_seq *x = (t_alien_seq *)pd_new(alien_seq_class);
    
    x->x_val_out = outlet_new(&x->x_obj, &s_anything);
    x->x_loop_out = outlet_new(&x->x_obj, &s_bang);
    x->x_step_out = outlet_new(&x->x_obj, &s_float);
    
    // Second inlet for step position
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("step"));
    
    x->x_max_steps = (max_steps > 0) ? (int)max_steps : 16;
    x->x_length = 0;
    x->x_step = 0;
    
    return (void *)x;
}

void alien_seq_setup(void) {
    alien_seq_class = class_new(gensym("alien_seq"),
        (t_newmethod)alien_seq_new,
        0,
        sizeof(t_alien_seq),
        CLASS_DEFAULT,
        A_DEFFLOAT,
        0);
    
    class_addbang(alien_seq_class, alien_seq_bang);
    class_addlist(alien_seq_class, alien_seq_list);
    class_addfloat(alien_seq_class, alien_seq_float);
    class_addsymbol(alien_seq_class, alien_seq_symbol);
    class_addmethod(alien_seq_class, (t_method)alien_seq_reset, gensym("reset"), 0);
    class_addmethod(alien_seq_class, (t_method)alien_seq_set_step, gensym("step"), A_FLOAT, 0);
    
    post("alien_seq - step sequencer for alien framework");
}
