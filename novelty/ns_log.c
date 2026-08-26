/*
 * ns_log - Append-writer for CSV / JSONL files for Pure Data
 *
 *   [ns_log run01.csv]            CSV (default)
 *   [ns_log run01.jsonl jsonl]    JSONL (one row per line)
 *
 * Hot left inlet (list / float / symbol):
 *     One row. Floats and symbols are written as-is. After writing,
 *     a bang is emitted on the outlet so a patch can chain logging
 *     to other actions.
 *
 * Cold right inlet (proxy):
 *     open <symbol>          open file (closes the current one). If the
 *                            path doesn't exist it is created; if it does
 *                            exist, rows are appended.
 *     close                  close the file
 *     flush                  flush buffered writes to disk
 *     header <list>          (CSV only) write a header row. Errors if the
 *                            file already has rows.
 *     tag <symbol>           prepend this token as the first column of
 *                            every subsequent row. Pass `tag` with no
 *                            arguments to clear.
 *     format <symbol>        csv | jsonl
 *
 * Outlet:
 *     left:  bang            after each successful row write / flush
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"

#include <string.h>
#include <stdio.h>

/* ======================================================================== */

static t_class *ns_log_class;
static t_class *ns_log_proxy_class;

typedef struct _ns_log t_ns_log;

typedef struct _ns_log_proxy {
    t_pd p_pd;
    t_ns_log *p_owner;
} t_ns_log_proxy;

typedef enum {
    NS_LOG_CSV = 0,
    NS_LOG_JSONL = 1,
} ns_log_format_t;

#define NS_LOG_TAG_MAX 64
#define NS_LOG_PATH_MAX 1024

struct _ns_log {
    t_object x_obj;
    t_outlet *x_out;
    FILE *x_file;
    ns_log_format_t x_format;
    int x_row_count;
    int x_has_tag;
    char x_tag[NS_LOG_TAG_MAX];
    char x_path[NS_LOG_PATH_MAX];
    t_ns_log_proxy x_proxy;
};

/* ======================================================================== */
/* HELPERS                                                                  */
/* ======================================================================== */

static void ns_log_close(t_ns_log *x) {
    if (x->x_file) {
        fclose(x->x_file);
        x->x_file = NULL;
        x->x_row_count = 0;
    }
}

static int ns_log_open(t_ns_log *x, const char *path) {
    ns_log_close(x);
    x->x_file = fopen(path, "a");
    if (!x->x_file) return 0;
    /* Probe the existing length so we know whether a header is allowed. */
    fseek(x->x_file, 0, SEEK_END);
    long sz = ftell(x->x_file);
    x->x_row_count = (sz > 0) ? -1 : 0;  /* -1 = unknown but non-empty */
    /* Remember the path so `truncate` can wipe the same file later. */
    snprintf(x->x_path, NS_LOG_PATH_MAX, "%s", path);
    return 1;
}

/* Write one atom in CSV form (no quoting heuristics — we expect numeric/symbol). */
static void write_atom_csv(FILE *f, const t_atom *a) {
    if (a->a_type == A_FLOAT) {
        double v = atom_getfloat((t_atom *)a);
        if (v == (double)(long long)v) fprintf(f, "%lld", (long long)v);
        else fprintf(f, "%.9g", v);
    } else if (a->a_type == A_SYMBOL) {
        fprintf(f, "%s", atom_getsymbol((t_atom *)a)->s_name);
    } else {
        fprintf(f, "");
    }
}

static void write_atom_json(FILE *f, const t_atom *a) {
    if (a->a_type == A_FLOAT) {
        double v = atom_getfloat((t_atom *)a);
        if (v == (double)(long long)v) fprintf(f, "%lld", (long long)v);
        else fprintf(f, "%.9g", v);
    } else if (a->a_type == A_SYMBOL) {
        const char *s = atom_getsymbol((t_atom *)a)->s_name;
        fputc('"', f);
        for (const char *p = s; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
            else if (c == '\n') { fputc('\\', f); fputc('n', f); }
            else if (c < 0x20) fprintf(f, "\\u%04x", c);
            else fputc(c, f);
        }
        fputc('"', f);
    } else {
        fprintf(f, "null");
    }
}

static void write_row(t_ns_log *x, t_symbol *selector, int argc, t_atom *argv) {
    if (!x->x_file) {
        pd_error(x, "ns_log: no open file (use 'open <path>')");
        return;
    }
    /* Optional selector handling — if it's a non-list selector, prepend it. */
    int has_sel = (selector
                   && selector != &s_list
                   && selector != &s_float
                   && selector != &s_symbol
                   && selector != &s_bang
                   && selector->s_name && selector->s_name[0] != '\0');

    if (x->x_format == NS_LOG_CSV) {
        int first = 1;
        if (x->x_has_tag) {
            fprintf(x->x_file, "%s", x->x_tag);
            first = 0;
        }
        if (has_sel) {
            if (!first) fputc(',', x->x_file);
            fprintf(x->x_file, "%s", selector->s_name);
            first = 0;
        }
        for (int i = 0; i < argc; i++) {
            if (!first) fputc(',', x->x_file);
            write_atom_csv(x->x_file, &argv[i]);
            first = 0;
        }
        fputc('\n', x->x_file);
    } else {
        /* JSONL: one object per line: {"tag": "...", "values": [...]} */
        fputc('{', x->x_file);
        int wrote = 0;
        if (x->x_has_tag) {
            fprintf(x->x_file, "\"tag\":\"%s\"", x->x_tag);
            wrote = 1;
        }
        if (has_sel) {
            if (wrote) fputc(',', x->x_file);
            fprintf(x->x_file, "\"sel\":\"%s\"", selector->s_name);
            wrote = 1;
        }
        if (wrote) fputc(',', x->x_file);
        fprintf(x->x_file, "\"values\":[");
        for (int i = 0; i < argc; i++) {
            if (i > 0) fputc(',', x->x_file);
            write_atom_json(x->x_file, &argv[i]);
        }
        fprintf(x->x_file, "]}\n");
    }
    x->x_row_count = (x->x_row_count < 0) ? -1 : x->x_row_count + 1;
    outlet_bang(x->x_out);
}

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

static void ns_log_anything(t_ns_log *x, t_symbol *s, int argc, t_atom *argv) {
    write_row(x, s, argc, argv);
}

static void ns_log_list(t_ns_log *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    write_row(x, &s_list, argc, argv);
}

static void ns_log_float(t_ns_log *x, t_floatarg f) {
    t_atom a;
    SETFLOAT(&a, f);
    write_row(x, &s_list, 1, &a);
}

static void ns_log_symbol(t_ns_log *x, t_symbol *s) {
    t_atom a;
    SETSYMBOL(&a, s);
    write_row(x, &s_list, 1, &a);
}

/* ======================================================================== */
/* PROXY (right inlet)                                                      */
/* ======================================================================== */

static void ns_log_proxy_open(t_ns_log_proxy *p, t_symbol *s) {
    if (!s || !s->s_name || s->s_name[0] == '\0') {
        pd_error(p->p_owner, "ns_log: open needs a path");
        return;
    }
    if (!ns_log_open(p->p_owner, s->s_name)) {
        pd_error(p->p_owner, "ns_log: failed to open %s", s->s_name);
    } else {
        post("ns_log: writing to %s (%s)", s->s_name,
             p->p_owner->x_format == NS_LOG_CSV ? "csv" : "jsonl");
    }
}

static void ns_log_proxy_close(t_ns_log_proxy *p) {
    ns_log_close(p->p_owner);
}

static void ns_log_proxy_flush(t_ns_log_proxy *p) {
    if (p->p_owner->x_file) {
        fflush(p->p_owner->x_file);
        outlet_bang(p->p_owner->x_out);
    }
}

/* Truncate the currently-open file to zero length, then re-open it for
 * append. Useful for [loadbang]-driven session setup so a fresh log starts
 * each time the patch is loaded, regardless of leftover rows from prior
 * sessions. */
static void ns_log_proxy_truncate(t_ns_log_proxy *p) {
    t_ns_log *x = p->p_owner;
    if (x->x_path[0] == '\0') {
        pd_error(x, "ns_log: truncate needs a previously-opened path");
        return;
    }
    /* Close, wipe via "w" mode, close, reopen as "a" — preserves invariants. */
    ns_log_close(x);
    FILE *wipe = fopen(x->x_path, "w");
    if (!wipe) { pd_error(x, "ns_log: truncate failed to wipe %s", x->x_path); return; }
    fclose(wipe);
    if (!ns_log_open(x, x->x_path)) {
        pd_error(x, "ns_log: truncate failed to reopen %s", x->x_path);
    }
}

static void ns_log_proxy_header(t_ns_log_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_log *x = p->p_owner;
    if (!x->x_file) { pd_error(x, "ns_log: header needs an open file"); return; }
    if (x->x_format != NS_LOG_CSV) {
        pd_error(x, "ns_log: header is CSV-only");
        return;
    }
    if (x->x_row_count != 0) {
        pd_error(x, "ns_log: header must come before any rows");
        return;
    }
    /* Writing tag column too if active. */
    int first = 1;
    if (x->x_has_tag) {
        fprintf(x->x_file, "tag");
        first = 0;
    }
    for (int i = 0; i < argc; i++) {
        if (!first) fputc(',', x->x_file);
        if (argv[i].a_type == A_SYMBOL) {
            fprintf(x->x_file, "%s", atom_getsymbol(&argv[i])->s_name);
        } else if (argv[i].a_type == A_FLOAT) {
            fprintf(x->x_file, "%g", atom_getfloat(&argv[i]));
        }
        first = 0;
    }
    fputc('\n', x->x_file);
    /* The header row itself shouldn't count as a data row. */
}

static void ns_log_proxy_tag(t_ns_log_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_log *x = p->p_owner;
    if (argc == 0) {
        x->x_has_tag = 0;
        x->x_tag[0] = '\0';
        return;
    }
    if (argv[0].a_type == A_SYMBOL) {
        const char *name = atom_getsymbol(&argv[0])->s_name;
        snprintf(x->x_tag, NS_LOG_TAG_MAX, "%s", name);
        x->x_has_tag = 1;
    } else if (argv[0].a_type == A_FLOAT) {
        snprintf(x->x_tag, NS_LOG_TAG_MAX, "%g", atom_getfloat(&argv[0]));
        x->x_has_tag = 1;
    }
}

static void ns_log_proxy_format(t_ns_log_proxy *p, t_symbol *s) {
    if (!s || !s->s_name) return;
    if (strcmp(s->s_name, "csv") == 0)         p->p_owner->x_format = NS_LOG_CSV;
    else if (strcmp(s->s_name, "jsonl") == 0)  p->p_owner->x_format = NS_LOG_JSONL;
    else pd_error(p->p_owner, "ns_log: unknown format '%s' (use csv|jsonl)", s->s_name);
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                 */
/* ======================================================================== */

static void *ns_log_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_log *x = (t_ns_log *)pd_new(ns_log_class);

    x->x_file = NULL;
    x->x_format = NS_LOG_CSV;
    x->x_row_count = 0;
    x->x_has_tag = 0;
    x->x_tag[0] = '\0';
    x->x_path[0] = '\0';

    /* Optional creation args: <path> [<format>] */
    t_symbol *path = NULL;
    if (argc > 0 && argv[0].a_type == A_SYMBOL) {
        path = atom_getsymbol(&argv[0]);
    }
    if (argc > 1 && argv[1].a_type == A_SYMBOL) {
        const char *fmt = atom_getsymbol(&argv[1])->s_name;
        if (strcmp(fmt, "jsonl") == 0) x->x_format = NS_LOG_JSONL;
        /* else keep csv default */
    } else if (path) {
        /* Detect format from extension. */
        const char *p = strrchr(path->s_name, '.');
        if (p && (strcmp(p, ".jsonl") == 0 || strcmp(p, ".json") == 0)) {
            x->x_format = NS_LOG_JSONL;
        }
    }

    /* Proxy + outlet */
    x->x_proxy.p_pd = ns_log_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);
    x->x_out = outlet_new(&x->x_obj, &s_bang);

    if (path && path->s_name[0] != '\0') {
        if (!ns_log_open(x, path->s_name)) {
            pd_error(x, "ns_log: failed to open %s", path->s_name);
        }
    }

    return (void *)x;
}

static void ns_log_free(t_ns_log *x) {
    ns_log_close(x);
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_log_setup(void) {
    ns_log_proxy_class = class_new(gensym("_ns_log_proxy"),
        0, 0, sizeof(t_ns_log_proxy), CLASS_PD, 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_open,
                    gensym("open"), A_SYMBOL, 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_close,
                    gensym("close"), 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_flush,
                    gensym("flush"), 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_truncate,
                    gensym("truncate"), 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_header,
                    gensym("header"), A_GIMME, 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_tag,
                    gensym("tag"), A_GIMME, 0);
    class_addmethod(ns_log_proxy_class, (t_method)ns_log_proxy_format,
                    gensym("format"), A_SYMBOL, 0);

    ns_log_class = class_new(gensym("ns_log"),
        (t_newmethod)ns_log_new,
        (t_method)ns_log_free,
        sizeof(t_ns_log),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addanything(ns_log_class, ns_log_anything);
    class_addlist(ns_log_class, ns_log_list);
    class_addfloat(ns_log_class, ns_log_float);
    class_addsymbol(ns_log_class, ns_log_symbol);

    post("ns_log %s - structured CSV/JSONL append writer", NS_VERSION_STRING);
}
