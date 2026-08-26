/*
 * ns_archive - Vector archive with k-NN novelty scoring for Pure Data
 *
 *   [ns_archive]              anonymous, instance-local storage
 *   [ns_archive shared]       named; all [ns_archive shared] in any patch
 *                             share a single underlying archive
 *
 * Hot left inlet (list):
 *     A behavioural-characterisation vector. Outputs the mean kNN
 *     distance to the current archive contents on the left outlet,
 *     then admits the vector. Archive size goes to the right outlet.
 *     If the archive is empty, the score is +inf (printed; the float
 *     outlet emits a very large number).
 *
 * Cold right inlet (proxy, accepts control messages):
 *     query <list>     score WITHOUT admitting
 *     clear            wipe all entries
 *     save <symbol>    persist to file (binary)
 *     load <symbol>    restore from file
 *     k <int>          set kNN k (0 = auto, default = min(15, count/3))
 *     metric <symbol>  l2 | cosine | hamming
 *     bang             output the current size to the right outlet
 *
 * Outlets:
 *     left:  float    novelty score (mean kNN distance)
 *     right: float    archive size after admission
 *
 * The first list received locks the archive's dimensionality.
 * Subsequent lists with mismatched length are rejected with an error.
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>
#include <stdio.h>

/* ======================================================================== */
/* CLASSES                                                                  */
/* ======================================================================== */

static t_class *ns_archive_class;
static t_class *ns_archive_proxy_class;

typedef struct _ns_archive t_ns_archive;

typedef struct _ns_archive_proxy {
    t_pd p_pd;
    t_ns_archive *p_owner;
} t_ns_archive_proxy;

struct _ns_archive {
    t_object x_obj;
    t_outlet *x_score_out;
    t_outlet *x_size_out;
    t_symbol *x_name;       /* NULL = anonymous */
    ns_archive_t *x_archive;
    t_ns_archive_proxy x_proxy;
};

/* ======================================================================== */
/* NAMED-INSTANCE REGISTRY                                                  */
/*                                                                          */
/* Multiple [ns_archive foo] objects share a single ns_archive_t,           */
/* refcounted. Anonymous instances each own their archive.                  */
/* ======================================================================== */

typedef struct ns_archive_entry {
    t_symbol *name;
    ns_archive_t *archive;
    struct ns_archive_entry *next;
} ns_archive_entry_t;

static ns_archive_entry_t *g_archive_registry = NULL;

static ns_archive_t *registry_acquire(t_symbol *name) {
    if (!name) return NULL;
    for (ns_archive_entry_t *e = g_archive_registry; e; e = e->next) {
        if (e->name == name) {
            e->archive->refcount++;
            return e->archive;
        }
    }
    ns_archive_t *a = ns_archive_create();
    if (!a) return NULL;
    ns_archive_entry_t *e = (ns_archive_entry_t *)getbytes(sizeof(ns_archive_entry_t));
    if (!e) { ns_archive_destroy(a); return NULL; }
    e->name = name;
    e->archive = a;
    e->next = g_archive_registry;
    g_archive_registry = e;
    return a;
}

static void registry_release(ns_archive_t *a) {
    if (!a) return;
    a->refcount--;
    if (a->refcount > 0) return;
    ns_archive_entry_t **pp = &g_archive_registry;
    while (*pp) {
        if ((*pp)->archive == a) {
            ns_archive_entry_t *dead = *pp;
            *pp = (*pp)->next;
            freebytes(dead, sizeof(ns_archive_entry_t));
            break;
        }
        pp = &(*pp)->next;
    }
    ns_archive_destroy(a);
}

/* ======================================================================== */
/* HELPERS                                                                  */
/* ======================================================================== */

#define NS_LIST_STACK 256

/* Convert an atom list to a float buffer. Returns dim, or -1 on error.
 * Caller-supplied buf must have at least argc slots. */
static int atoms_to_floats(int argc, t_atom *argv, float *buf, int max) {
    if (argc > max) return -1;
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type != A_FLOAT) return -1;
        buf[i] = (float)atom_getfloat(&argv[i]);
    }
    return argc;
}

static void ns_archive_emit_size(t_ns_archive *x) {
    outlet_float(x->x_size_out, (t_float)x->x_archive->count);
}

/* ======================================================================== */
/* HOT LEFT INLET — score AND admit                                         */
/* ======================================================================== */

static void ns_archive_list(t_ns_archive *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc < 1) {
        pd_error(x, "ns_archive: empty list");
        return;
    }
    if (argc > NS_MAX_DIM) {
        pd_error(x, "ns_archive: list length %d exceeds NS_MAX_DIM (%d)", argc, NS_MAX_DIM);
        return;
    }
    /* Use stack buffer for typical sizes; fall back to heap for big BCs. */
    float stack_buf[NS_LIST_STACK];
    float *buf = stack_buf;
    int allocated = 0;
    if (argc > NS_LIST_STACK) {
        buf = (float *)getbytes(sizeof(float) * argc);
        if (!buf) { pd_error(x, "ns_archive: out of memory"); return; }
        allocated = 1;
    }

    int dim = atoms_to_floats(argc, argv, buf, argc);
    if (dim < 0) {
        pd_error(x, "ns_archive: list must contain only floats");
        if (allocated) freebytes(buf, sizeof(float) * argc);
        return;
    }

    /* dim check happens implicitly inside ns_archive_score / add. */
    if (x->x_archive->dim != 0 && x->x_archive->dim != dim) {
        pd_error(x, "ns_archive: dim mismatch (have %d, got %d)",
                 x->x_archive->dim, dim);
        if (allocated) freebytes(buf, sizeof(float) * argc);
        return;
    }

    float score = ns_archive_score(x->x_archive, buf, dim);
    int admitted = ns_archive_add(x->x_archive, buf, dim);

    if (allocated) freebytes(buf, sizeof(float) * argc);

    if (!admitted) {
        pd_error(x, "ns_archive: admit failed (oom or dim mismatch)");
        return;
    }

    /* Emit size first (right-to-left convention), then score (left/hot). */
    ns_archive_emit_size(x);
    /* Pd's float message can't carry +inf cleanly; substitute a large value. */
    if (!isfinite(score)) score = 1e30f;
    outlet_float(x->x_score_out, (t_float)score);
}

/* ======================================================================== */
/* PROXY (right inlet) — control messages                                   */
/* ======================================================================== */

static void ns_archive_proxy_query(t_ns_archive_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_archive *x = p->p_owner;
    if (argc < 1) { pd_error(x, "ns_archive: query needs a list"); return; }

    float stack_buf[NS_LIST_STACK];
    float *buf = stack_buf;
    int allocated = 0;
    if (argc > NS_LIST_STACK) {
        buf = (float *)getbytes(sizeof(float) * argc);
        if (!buf) { pd_error(x, "ns_archive: out of memory"); return; }
        allocated = 1;
    }
    int dim = atoms_to_floats(argc, argv, buf, argc);
    if (dim < 0) {
        pd_error(x, "ns_archive: query list must be all floats");
        if (allocated) freebytes(buf, sizeof(float) * argc);
        return;
    }
    if (x->x_archive->dim != 0 && x->x_archive->dim != dim) {
        pd_error(x, "ns_archive: query dim mismatch (have %d, got %d)",
                 x->x_archive->dim, dim);
        if (allocated) freebytes(buf, sizeof(float) * argc);
        return;
    }
    float score = ns_archive_score(x->x_archive, buf, dim);
    if (allocated) freebytes(buf, sizeof(float) * argc);
    if (!isfinite(score)) score = 1e30f;
    outlet_float(x->x_score_out, (t_float)score);
}

static void ns_archive_proxy_clear(t_ns_archive_proxy *p) {
    ns_archive_clear(p->p_owner->x_archive);
    ns_archive_emit_size(p->p_owner);
}

static void ns_archive_proxy_save(t_ns_archive_proxy *p, t_symbol *s) {
    if (!s || !s->s_name || s->s_name[0] == '\0') {
        pd_error(p->p_owner, "ns_archive: save needs a path");
        return;
    }
    if (!ns_archive_save(p->p_owner->x_archive, s->s_name)) {
        pd_error(p->p_owner, "ns_archive: save failed (path: %s)", s->s_name);
    } else {
        post("ns_archive: saved %d vectors to %s",
             p->p_owner->x_archive->count, s->s_name);
    }
}

static void ns_archive_proxy_load(t_ns_archive_proxy *p, t_symbol *s) {
    if (!s || !s->s_name || s->s_name[0] == '\0') {
        pd_error(p->p_owner, "ns_archive: load needs a path");
        return;
    }
    if (!ns_archive_load(p->p_owner->x_archive, s->s_name)) {
        pd_error(p->p_owner, "ns_archive: load failed (path: %s)", s->s_name);
    } else {
        post("ns_archive: loaded %d vectors from %s",
             p->p_owner->x_archive->count, s->s_name);
        ns_archive_emit_size(p->p_owner);
    }
}

static void ns_archive_proxy_k(t_ns_archive_proxy *p, t_floatarg f) {
    int k = (int)f;
    if (k < 0) k = 0;
    p->p_owner->x_archive->k = k;
}

static void ns_archive_proxy_metric(t_ns_archive_proxy *p, t_symbol *s) {
    if (!s || !s->s_name) return;
    if (strcmp(s->s_name, "l2") == 0)        p->p_owner->x_archive->metric = NS_DIST_L2;
    else if (strcmp(s->s_name, "cosine") == 0)  p->p_owner->x_archive->metric = NS_DIST_COSINE;
    else if (strcmp(s->s_name, "hamming") == 0) p->p_owner->x_archive->metric = NS_DIST_HAMMING;
    else {
        pd_error(p->p_owner, "ns_archive: unknown metric '%s' (use l2|cosine|hamming)", s->s_name);
    }
}

static void ns_archive_proxy_bang(t_ns_archive_proxy *p) {
    ns_archive_emit_size(p->p_owner);
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                 */
/* ======================================================================== */

static void *ns_archive_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_archive *x = (t_ns_archive *)pd_new(ns_archive_class);

    /* Optional first arg: name (symbol) → shared/named instance. */
    x->x_name = NULL;
    if (argc > 0 && argv[0].a_type == A_SYMBOL) {
        x->x_name = atom_getsymbol(&argv[0]);
        x->x_archive = registry_acquire(x->x_name);
    } else {
        x->x_archive = ns_archive_create();
    }
    if (!x->x_archive) {
        pd_error(x, "ns_archive: failed to create archive");
        return NULL;
    }

    /* Proxy for right inlet. */
    x->x_proxy.p_pd = ns_archive_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlets: score (float), size (float). */
    x->x_score_out = outlet_new(&x->x_obj, &s_float);
    x->x_size_out = outlet_new(&x->x_obj, &s_float);

    return (void *)x;
}

static void ns_archive_free(t_ns_archive *x) {
    if (x->x_name) {
        registry_release(x->x_archive);
    } else {
        ns_archive_destroy(x->x_archive);
    }
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_archive_setup(void) {
    /* Proxy class for the right (cold) inlet. */
    ns_archive_proxy_class = class_new(gensym("_ns_archive_proxy"),
        0, 0, sizeof(t_ns_archive_proxy), CLASS_PD, 0);
    class_addmethod(ns_archive_proxy_class, (t_method)ns_archive_proxy_query,
                    gensym("query"), A_GIMME, 0);
    class_addmethod(ns_archive_proxy_class, (t_method)ns_archive_proxy_clear,
                    gensym("clear"), 0);
    class_addmethod(ns_archive_proxy_class, (t_method)ns_archive_proxy_save,
                    gensym("save"), A_SYMBOL, 0);
    class_addmethod(ns_archive_proxy_class, (t_method)ns_archive_proxy_load,
                    gensym("load"), A_SYMBOL, 0);
    class_addmethod(ns_archive_proxy_class, (t_method)ns_archive_proxy_k,
                    gensym("k"), A_FLOAT, 0);
    class_addmethod(ns_archive_proxy_class, (t_method)ns_archive_proxy_metric,
                    gensym("metric"), A_SYMBOL, 0);
    class_addbang(ns_archive_proxy_class, ns_archive_proxy_bang);

    /* Main class. */
    ns_archive_class = class_new(gensym("ns_archive"),
        (t_newmethod)ns_archive_new,
        (t_method)ns_archive_free,
        sizeof(t_ns_archive),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addlist(ns_archive_class, ns_archive_list);

    post("ns_archive %s - vector archive with kNN novelty scoring", NS_VERSION_STRING);
}
