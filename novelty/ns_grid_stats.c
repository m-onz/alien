/*
 * ns_grid_stats - 2D grid → 6-dim feature vector for Pure Data
 *
 *   [ns_grid_stats 4 4]      a 4×4 grid (16 cells flat)
 *   [ns_grid_stats 16 16]
 *
 * Hot left inlet (list):
 *     A flat list of W*H floats. Each cell is treated as binary:
 *     >= 0.5 → live, < 0.5 → dead. The output is always a list of
 *     6 floats, in this order:
 *
 *         0  density           mean of cells, in [0, 1]
 *         1  spatial_entropy   Shannon entropy of 2×2 sub-block densities,
 *                              normalized to [0, 1]
 *         2  sym_h             1 - mean|top - flipped(bottom)|
 *         3  sym_v             1 - mean|left - flipped(right)|
 *         4  components        4-connected component count, normalized
 *         5  mean_comp_size    mean component size / total cells
 *
 * Cold right inlet:
 *     resize <int> <int>      change W and H. Subsequent lists must
 *                             match the new W*H length.
 *
 * Outlet:
 *     left:  list             6 floats (always)
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>

/* ======================================================================== */

static t_class *ns_grid_stats_class;
static t_class *ns_grid_stats_proxy_class;

typedef struct _ns_grid_stats t_ns_grid_stats;

typedef struct _ns_grid_stats_proxy {
    t_pd p_pd;
    t_ns_grid_stats *p_owner;
} t_ns_grid_stats_proxy;

#define NS_GRID_LIST_STACK 1024

struct _ns_grid_stats {
    t_object x_obj;
    t_outlet *x_out;
    int x_w;
    int x_h;
    t_ns_grid_stats_proxy x_proxy;
};

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_grid_stats_list(t_ns_grid_stats *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    int expected = x->x_w * x->x_h;
    if (argc != expected) {
        pd_error(x, "ns_grid_stats: expected %d cells (%dx%d), got %d",
                 expected, x->x_w, x->x_h, argc);
        return;
    }
    if (x->x_w <= 0 || x->x_h <= 0 || x->x_w > NS_GRID_MAX || x->x_h > NS_GRID_MAX) {
        pd_error(x, "ns_grid_stats: invalid dimensions %dx%d (max %d)",
                 x->x_w, x->x_h, NS_GRID_MAX);
        return;
    }

    float stack_buf[NS_GRID_LIST_STACK];
    float *grid = stack_buf;
    int allocated = 0;
    if (argc > NS_GRID_LIST_STACK) {
        grid = (float *)getbytes(sizeof(float) * argc);
        if (!grid) { pd_error(x, "ns_grid_stats: out of memory"); return; }
        allocated = 1;
    }
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type != A_FLOAT) {
            pd_error(x, "ns_grid_stats: list must be all floats");
            if (allocated) freebytes(grid, sizeof(float) * argc);
            return;
        }
        grid[i] = (float)atom_getfloat(&argv[i]);
    }

    float out6[6];
    ns_grid_stats(grid, x->x_w, x->x_h, out6);

    if (allocated) freebytes(grid, sizeof(float) * argc);

    t_atom out_atoms[6];
    for (int i = 0; i < 6; i++) SETFLOAT(&out_atoms[i], (t_float)out6[i]);
    outlet_list(x->x_out, &s_list, 6, out_atoms);
}

/* ======================================================================== */
/* PROXY                                                                    */
/* ======================================================================== */

static void ns_grid_stats_proxy_resize(t_ns_grid_stats_proxy *p, t_floatarg fw, t_floatarg fh) {
    int w = (int)fw, h = (int)fh;
    if (w <= 0 || h <= 0 || w > NS_GRID_MAX || h > NS_GRID_MAX) {
        pd_error(p->p_owner, "ns_grid_stats: bad dims %dx%d (max %d)", w, h, NS_GRID_MAX);
        return;
    }
    p->p_owner->x_w = w;
    p->p_owner->x_h = h;
}

/* ======================================================================== */
/* CONSTRUCTOR                                                              */
/* ======================================================================== */

static void *ns_grid_stats_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_grid_stats *x = (t_ns_grid_stats *)pd_new(ns_grid_stats_class);

    /* Required: w, h. Default to 4×4. */
    int w = 4, h = 4;
    if (argc >= 2 && argv[0].a_type == A_FLOAT && argv[1].a_type == A_FLOAT) {
        w = (int)atom_getfloat(&argv[0]);
        h = (int)atom_getfloat(&argv[1]);
    }
    if (w <= 0 || h <= 0 || w > NS_GRID_MAX || h > NS_GRID_MAX) {
        pd_error(x, "ns_grid_stats: bad dims (using 4x4)");
        w = 4; h = 4;
    }
    x->x_w = w;
    x->x_h = h;

    x->x_proxy.p_pd = ns_grid_stats_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    x->x_out = outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_grid_stats_setup(void) {
    ns_grid_stats_proxy_class = class_new(gensym("_ns_grid_stats_proxy"),
        0, 0, sizeof(t_ns_grid_stats_proxy), CLASS_PD, 0);
    class_addmethod(ns_grid_stats_proxy_class, (t_method)ns_grid_stats_proxy_resize,
                    gensym("resize"), A_FLOAT, A_FLOAT, 0);

    ns_grid_stats_class = class_new(gensym("ns_grid_stats"),
        (t_newmethod)ns_grid_stats_new,
        0,
        sizeof(t_ns_grid_stats),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addlist(ns_grid_stats_class, ns_grid_stats_list);

    post("ns_grid_stats %s - 2D grid → 6-dim BC", NS_VERSION_STRING);
}
