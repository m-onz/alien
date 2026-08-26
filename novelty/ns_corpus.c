/*
 * ns_corpus - Rolling corpus of validated alien DSL expressions
 *
 *   [ns_corpus]                anonymous, instance-local corpus
 *   [ns_corpus pool]           named; all [ns_corpus pool] share state
 *   [ns_corpus pool 256]       named, with capacity
 *
 * The Producer-side memory the existing framework was missing. Where
 * ns_archive owns BC novelty scoring (semantic memory of "where in
 * feature space have I been"), ns_corpus owns parent selection (a
 * pool of admission-passed expressions to draw from). Both are needed
 * for a population-based GA — they answer different questions at
 * different points in the pipeline.
 *
 * Hot left inlet:
 *     bang             sample one parent → left outlet
 *     <expr>           STAGE candidate. Cached for the next admit/
 *                      add/protect message. Doesn't emit anything.
 *
 * Cold right inlet (proxy) — admission control:
 *     admit <novelty> <quality>
 *                      finalise the staged candidate. Subject to dedup,
 *                      min_quality, min_novelty gates. Emits 1 or 0 on
 *                      the right outlet plus an event log message.
 *     add              unconditional add of staged (still dedup-checked)
 *     protect          unconditional add, marked non-evictable
 *     pair             sample two parents; emits a single
 *                      `crossover <expr_a> <expr_b>` message on the
 *                      LEFT outlet so it can wire straight to
 *                      ns_seq_propose's right inlet.
 *
 * Cold right inlet — config / persistence:
 *     seeds <path>     load text file; each parse-validating line
 *                      becomes a protected entry. Posts a summary.
 *     save <path>      binary persistence (full metadata)
 *     load <path>      restore from binary
 *     cap <n>          maximum entries (default 256)
 *     min_novelty <f>  admission threshold θ (default 0.0)
 *     min_quality <f>  admission threshold (default 0.3)
 *     selection <s>    sampling: uniform | weighted | tournament
 *     seed <int>       deterministic RNG seed
 *     clear            drop all non-protected entries
 *     dump             emit every entry on the right outlet
 *     size             emit current count on the right outlet
 *
 * Outlets:
 *     left   anything  sampled parent expression (bang) or
 *                      `crossover <a> <b>` (pair)
 *     right  anything  event stream — `admit 1 <expr>`, `admit 0 <expr>
 *                      <reason>`, `evict <expr>`, `sample <expr>`,
 *                      `size <n>`, `dump <expr> <nov> <qual> <gen>`
 */

#define PD 1
#include "m_pd.h"
#include "ns_core.h"
#include "../alien_core.h"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* ======================================================================== */
/* DATA STRUCTURES                                                          */
/* ======================================================================== */

#define NS_CORPUS_EXPR_MAX 1024
#define NS_CORPUS_DEFAULT_CAP 256
#define NS_CORPUS_FILE_MAGIC 0x4E435250u   /* "NCRP" */
#define NS_CORPUS_FILE_VERSION 1u
#define NS_CORPUS_AGE_WINDOW_SEC 3600.0f   /* recency bonus tapers over 1h */
#define NS_CORPUS_AGE_BONUS 0.1f

typedef enum {
    NS_CORPUS_SEL_UNIFORM = 0,
    NS_CORPUS_SEL_WEIGHTED = 1,
    NS_CORPUS_SEL_TOURNAMENT = 2,
} ns_corpus_selection_t;

typedef struct {
    char expr[NS_CORPUS_EXPR_MAX];
    float novelty;
    float quality;
    int generation;
    uint32_t hash;            /* FNV-1a of expr, for dedup */
    uint32_t timestamp;       /* admission time, seconds since corpus init */
    uint8_t is_protected;     /* 1 = never evict */
    uint8_t reserved[3];
} ns_corpus_entry_t;

typedef struct {
    ns_corpus_entry_t *entries;
    int count;
    int capacity;             /* allocated capacity (in entries) */
    int max_entries;          /* user-set cap */
    float min_novelty;
    float min_quality;
    ns_corpus_selection_t selection;
    int generation_counter;
    ns_rng_t rng;
    uint32_t corpus_start_time;  /* for relative timestamps */
    int refcount;
} ns_corpus_t;

/* ======================================================================== */
/* RNG INTEGER RANGE                                                         */
/* ns_alien_ast.h defines ns_rng_int_range, but pulling in that whole        */
/* header just for one helper costs us its NS_OPS[] table and AST machinery */
/* we don't use here. Inline a local copy.                                   */
/* ======================================================================== */

static inline int corpus_rng_int_range(ns_rng_t *r, int lo, int hi) {
    if (hi < lo) return lo;
    int range = hi - lo + 1;
    return lo + (int)(ns_rng_uniform(r) * (float)range);
}

/* ======================================================================== */
/* HASHING & STRING UTILITIES                                                */
/* ======================================================================== */

static inline uint32_t ns_fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 16777619u;
    }
    return h;
}

/* In-place strip leading/trailing whitespace and trailing semicolon. */
static void normalize_line(char *s) {
    int n = (int)strlen(s);
    while (n > 0 && (isspace((unsigned char)s[n-1]) || s[n-1] == ';')) {
        s[--n] = '\0';
    }
    int i = 0;
    while (s[i] && isspace((unsigned char)s[i])) i++;
    if (i > 0) memmove(s, s + i, strlen(s + i) + 1);
}

/* True if a DSL string parses successfully. The render check is too
 * expensive at admission time — quality is checked separately by
 * ns_quality, which sees the rendered output. */
static int parse_validates(const char *expr) {
    if (!expr || expr[0] == '\0') return 0;
    Token tokens[2048];
    int n_tok = tokenize(expr, tokens, 2048);
    if (n_tok < 0) return 0;
    ASTNode *root = parse(tokens, n_tok);
    if (!root) return 0;
    ast_free(root);
    return 1;
}

/* ======================================================================== */
/* CORE OPERATIONS                                                           */
/* ======================================================================== */

static ns_corpus_t *ns_corpus_create(int cap) {
    ns_corpus_t *c = (ns_corpus_t *)getbytes(sizeof(ns_corpus_t));
    if (!c) return NULL;
    if (cap < 4) cap = NS_CORPUS_DEFAULT_CAP;
    c->entries = (ns_corpus_entry_t *)getbytes(sizeof(ns_corpus_entry_t) * cap);
    if (!c->entries) { freebytes(c, sizeof(ns_corpus_t)); return NULL; }
    c->count = 0;
    c->capacity = cap;
    c->max_entries = cap;
    c->min_novelty = 0.0f;
    c->min_quality = 0.3f;
    c->selection = NS_CORPUS_SEL_UNIFORM;
    c->generation_counter = 0;
    c->corpus_start_time = (uint32_t)time(NULL);
    c->refcount = 1;
    /* Seed RNG from time + pointer for non-determinism by default. */
    uint64_t seed = (uint64_t)time(NULL);
    seed ^= ((uint64_t)(uintptr_t)c) * 0x9E3779B97F4A7C15ULL;
    ns_rng_seed(&c->rng, seed);
    return c;
}

static void ns_corpus_destroy(ns_corpus_t *c) {
    if (!c) return;
    if (c->entries) freebytes(c->entries, sizeof(ns_corpus_entry_t) * c->capacity);
    freebytes(c, sizeof(ns_corpus_t));
}

/* Grow underlying storage if max_entries was raised past capacity. */
static int ns_corpus_reserve(ns_corpus_t *c, int new_cap) {
    if (c->capacity >= new_cap) return 1;
    int target = c->capacity > 0 ? c->capacity : 16;
    while (target < new_cap) target *= 2;
    size_t old_bytes = sizeof(ns_corpus_entry_t) * c->capacity;
    size_t new_bytes = sizeof(ns_corpus_entry_t) * target;
    ns_corpus_entry_t *p = (ns_corpus_entry_t *)resizebytes(c->entries, old_bytes, new_bytes);
    if (!p) return 0;
    c->entries = p;
    c->capacity = target;
    return 1;
}

/* Linear scan for hash match. Returns index or -1. */
static int ns_corpus_find_hash(const ns_corpus_t *c, uint32_t hash, const char *expr) {
    for (int i = 0; i < c->count; i++) {
        if (c->entries[i].hash == hash && strcmp(c->entries[i].expr, expr) == 0) {
            return i;
        }
    }
    return -1;
}

/* Eviction score — lower is more evictable. Recently-admitted entries
 * get a small bonus to give them a window before they can be evicted. */
static float entry_eviction_score(const ns_corpus_entry_t *e, uint32_t now,
                                  uint32_t corpus_start) {
    float fitness = e->novelty * e->quality;
    float age_sec = (float)(now - corpus_start - e->timestamp);
    if (age_sec < 0.0f) age_sec = 0.0f;
    float age_bonus = NS_CORPUS_AGE_BONUS *
        (1.0f - age_sec / NS_CORPUS_AGE_WINDOW_SEC);
    if (age_bonus < 0.0f) age_bonus = 0.0f;
    return fitness + age_bonus;
}

/* Find the most-evictable non-protected entry; returns index or -1. */
static int ns_corpus_pick_victim(const ns_corpus_t *c) {
    uint32_t now = (uint32_t)time(NULL);
    int best = -1;
    float best_score = 1e30f;
    for (int i = 0; i < c->count; i++) {
        if (c->entries[i].is_protected) continue;
        float s = entry_eviction_score(&c->entries[i], now, c->corpus_start_time);
        if (s < best_score) { best_score = s; best = i; }
    }
    return best;
}

/* Swap-remove at index. */
static void ns_corpus_remove(ns_corpus_t *c, int idx) {
    if (idx < 0 || idx >= c->count) return;
    int last = c->count - 1;
    if (idx != last) c->entries[idx] = c->entries[last];
    c->count = last;
}

/* ======================================================================== */
/* NAMED-INSTANCE REGISTRY (mirror ns_archive's pattern)                    */
/* ======================================================================== */

typedef struct ns_corpus_registry_entry {
    t_symbol *name;
    ns_corpus_t *corpus;
    struct ns_corpus_registry_entry *next;
} ns_corpus_registry_entry_t;

static ns_corpus_registry_entry_t *g_corpus_registry = NULL;

static ns_corpus_t *registry_acquire(t_symbol *name, int cap) {
    if (!name) return NULL;
    for (ns_corpus_registry_entry_t *e = g_corpus_registry; e; e = e->next) {
        if (e->name == name) {
            e->corpus->refcount++;
            return e->corpus;
        }
    }
    ns_corpus_t *c = ns_corpus_create(cap);
    if (!c) return NULL;
    ns_corpus_registry_entry_t *e =
        (ns_corpus_registry_entry_t *)getbytes(sizeof(*e));
    if (!e) { ns_corpus_destroy(c); return NULL; }
    e->name = name;
    e->corpus = c;
    e->next = g_corpus_registry;
    g_corpus_registry = e;
    return c;
}

static void registry_release(ns_corpus_t *c) {
    if (!c) return;
    c->refcount--;
    if (c->refcount > 0) return;
    ns_corpus_registry_entry_t **pp = &g_corpus_registry;
    while (*pp) {
        if ((*pp)->corpus == c) {
            ns_corpus_registry_entry_t *dead = *pp;
            *pp = (*pp)->next;
            freebytes(dead, sizeof(*dead));
            break;
        }
        pp = &(*pp)->next;
    }
    ns_corpus_destroy(c);
}

/* ======================================================================== */
/* PD CLASSES                                                                */
/* ======================================================================== */

static t_class *ns_corpus_class;
static t_class *ns_corpus_proxy_class;

typedef struct _ns_corpus t_ns_corpus;

typedef struct _ns_corpus_proxy {
    t_pd p_pd;
    t_ns_corpus *p_owner;
} t_ns_corpus_proxy;

struct _ns_corpus {
    t_object x_obj;
    t_outlet *x_out_left;     /* sampled parent / pair message */
    t_outlet *x_out_right;    /* event stream */
    t_symbol *x_name;         /* NULL = anonymous */
    ns_corpus_t *x_corpus;
    char x_staged[NS_CORPUS_EXPR_MAX];   /* per-instance staging */
    int x_has_staged;
    t_ns_corpus_proxy x_proxy;
};

/* ======================================================================== */
/* MESSAGE EMISSION HELPERS                                                  */
/* ======================================================================== */

static void emit_event_admit(t_ns_corpus *x, int ok, const char *expr, const char *reason) {
    if (ok) {
        t_atom args[2];
        SETFLOAT(&args[0], 1.0f);
        SETSYMBOL(&args[1], gensym(expr));
        outlet_anything(x->x_out_right, gensym("admit"), 2, args);
    } else {
        t_atom args[3];
        SETFLOAT(&args[0], 0.0f);
        SETSYMBOL(&args[1], gensym(expr));
        SETSYMBOL(&args[2], gensym(reason));
        outlet_anything(x->x_out_right, gensym("admit"), 3, args);
    }
}

static void emit_event_evict(t_ns_corpus *x, const char *expr) {
    t_atom a;
    SETSYMBOL(&a, gensym(expr));
    outlet_anything(x->x_out_right, gensym("evict"), 1, &a);
}

static void emit_event_sample(t_ns_corpus *x, const char *expr) {
    t_atom a;
    SETSYMBOL(&a, gensym(expr));
    outlet_anything(x->x_out_right, gensym("sample"), 1, &a);
}

static void emit_event_size(t_ns_corpus *x) {
    t_atom a;
    SETFLOAT(&a, (t_float)x->x_corpus->count);
    outlet_anything(x->x_out_right, gensym("size"), 1, &a);
}

static void emit_parent(t_ns_corpus *x, const char *expr) {
    /* Same convention as ns_seq_propose: the DSL string IS the symbol. */
    outlet_anything(x->x_out_left, gensym(expr), 0, NULL);
}

/* ======================================================================== */
/* SAMPLING                                                                  */
/* ======================================================================== */

static int sample_uniform(ns_corpus_t *c) {
    if (c->count <= 0) return -1;
    return corpus_rng_int_range(&c->rng, 0, c->count - 1);
}

static int sample_weighted(ns_corpus_t *c) {
    if (c->count <= 0) return -1;
    float total = 0.0f;
    for (int i = 0; i < c->count; i++) {
        float w = c->entries[i].novelty * c->entries[i].quality;
        if (w < 0.001f) w = 0.001f;  /* floor so freshly-added entries are reachable */
        total += w;
    }
    if (total <= 0.0f) return sample_uniform(c);
    float pick = ns_rng_uniform(&c->rng) * total;
    float acc = 0.0f;
    for (int i = 0; i < c->count; i++) {
        float w = c->entries[i].novelty * c->entries[i].quality;
        if (w < 0.001f) w = 0.001f;
        acc += w;
        if (pick <= acc) return i;
    }
    return c->count - 1;
}

static int sample_tournament(ns_corpus_t *c) {
    if (c->count <= 0) return -1;
    const int K = 3;
    int best = sample_uniform(c);
    float best_score = c->entries[best].novelty * c->entries[best].quality;
    for (int i = 1; i < K; i++) {
        int idx = sample_uniform(c);
        float s = c->entries[idx].novelty * c->entries[idx].quality;
        if (s > best_score) { best = idx; best_score = s; }
    }
    return best;
}

static int sample_one(ns_corpus_t *c) {
    if (c->count <= 0) return -1;
    switch (c->selection) {
        case NS_CORPUS_SEL_WEIGHTED:   return sample_weighted(c);
        case NS_CORPUS_SEL_TOURNAMENT: return sample_tournament(c);
        case NS_CORPUS_SEL_UNIFORM:
        default:                       return sample_uniform(c);
    }
}

/* ======================================================================== */
/* ADMISSION                                                                 */
/* ======================================================================== */

/* Internal admit. Returns 1 on admit, 0 on reject. The reason string
 * is filled for the event log when 0 is returned. */
static int do_admit(t_ns_corpus *x, const char *expr,
                    float novelty, float quality, int force, int is_protected,
                    const char **reject_reason) {
    ns_corpus_t *c = x->x_corpus;

    if (!parse_validates(expr)) {
        *reject_reason = "parse";
        return 0;
    }
    uint32_t hash = ns_fnv1a(expr);
    if (ns_corpus_find_hash(c, hash, expr) >= 0) {
        *reject_reason = "duplicate";
        return 0;
    }
    if (!force) {
        if (quality < c->min_quality) {
            *reject_reason = "quality";
            return 0;
        }
        if (novelty < c->min_novelty) {
            *reject_reason = "novelty";
            return 0;
        }
    }
    if (c->count >= c->max_entries) {
        int victim = ns_corpus_pick_victim(c);
        if (victim < 0) {
            *reject_reason = "full_protected";
            return 0;
        }
        char victim_expr[NS_CORPUS_EXPR_MAX];
        snprintf(victim_expr, NS_CORPUS_EXPR_MAX, "%s", c->entries[victim].expr);
        ns_corpus_remove(c, victim);
        emit_event_evict(x, victim_expr);
    }
    if (!ns_corpus_reserve(c, c->count + 1)) {
        *reject_reason = "oom";
        return 0;
    }
    ns_corpus_entry_t *e = &c->entries[c->count];
    snprintf(e->expr, NS_CORPUS_EXPR_MAX, "%s", expr);
    e->novelty = novelty;
    e->quality = quality;
    e->generation = ++c->generation_counter;
    e->hash = hash;
    e->timestamp = (uint32_t)time(NULL) - c->corpus_start_time;
    e->is_protected = (uint8_t)(is_protected ? 1 : 0);
    c->count++;
    return 1;
}

/* ======================================================================== */
/* HOT LEFT INLET                                                           */
/* ======================================================================== */

/* Reassemble Pd atom-form messages back into a DSL string.
 * Same logic as ns_seq_propose's atoms_to_dsl_string. */
static void atoms_to_dsl(t_symbol *s, int argc, t_atom *argv, char *buf, int max) {
    int pos = 0;
    if (s && s->s_name && s->s_name[0] != '\0' &&
        s != &s_list && s != &s_symbol && s != &s_float && s != &s_bang) {
        int w = snprintf(buf + pos, max - pos, "%s", s->s_name);
        if (w > 0 && w < max - pos) pos += w;
    }
    for (int i = 0; i < argc && pos < max - 1; i++) {
        if (pos > 0 && pos < max - 1) buf[pos++] = ' ';
        if (argv[i].a_type == A_FLOAT) {
            float f = atom_getfloat(&argv[i]);
            int w;
            if (f == (int)f) w = snprintf(buf + pos, max - pos, "%d", (int)f);
            else             w = snprintf(buf + pos, max - pos, "%g", f);
            if (w > 0 && w < max - pos) pos += w;
        } else if (argv[i].a_type == A_SYMBOL) {
            int w = snprintf(buf + pos, max - pos, "%s",
                             atom_getsymbol(&argv[i])->s_name);
            if (w > 0 && w < max - pos) pos += w;
        }
    }
    buf[pos < max ? pos : max - 1] = '\0';
}

/* bang on hot left → sample one parent. */
static void ns_corpus_bang(t_ns_corpus *x) {
    int idx = sample_one(x->x_corpus);
    if (idx < 0) {
        pd_error(x, "ns_corpus: empty — nothing to sample");
        return;
    }
    const char *expr = x->x_corpus->entries[idx].expr;
    emit_event_sample(x, expr);
    emit_parent(x, expr);
}

/* Anything other than bang on hot left → STAGE the expression. */
static void ns_corpus_anything(t_ns_corpus *x, t_symbol *s, int argc, t_atom *argv) {
    char buf[NS_CORPUS_EXPR_MAX];
    atoms_to_dsl(s, argc, argv, buf, NS_CORPUS_EXPR_MAX);
    if (buf[0] == '\0') return;
    snprintf(x->x_staged, NS_CORPUS_EXPR_MAX, "%s", buf);
    x->x_has_staged = 1;
}

static void ns_corpus_list(t_ns_corpus *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    ns_corpus_anything(x, &s_list, argc, argv);
}

static void ns_corpus_symbol(t_ns_corpus *x, t_symbol *s) {
    if (!s || !s->s_name || s->s_name[0] == '\0') return;
    snprintf(x->x_staged, NS_CORPUS_EXPR_MAX, "%s", s->s_name);
    x->x_has_staged = 1;
}

/* ======================================================================== */
/* PROXY (right inlet) — admission control                                  */
/* ======================================================================== */

static void proxy_admit(t_ns_corpus_proxy *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_corpus *x = p->p_owner;
    if (!x->x_has_staged) {
        pd_error(x, "ns_corpus: admit with no staged expression");
        return;
    }
    if (argc < 2 || argv[0].a_type != A_FLOAT || argv[1].a_type != A_FLOAT) {
        pd_error(x, "ns_corpus: admit needs <novelty> <quality>");
        return;
    }
    float novelty = atom_getfloat(&argv[0]);
    float quality = atom_getfloat(&argv[1]);
    const char *reason = NULL;
    int ok = do_admit(x, x->x_staged, novelty, quality, 0, 0, &reason);
    emit_event_admit(x, ok, x->x_staged, ok ? "" : (reason ? reason : "unknown"));
    x->x_has_staged = 0;
    x->x_staged[0] = '\0';
}

static void proxy_add(t_ns_corpus_proxy *p) {
    t_ns_corpus *x = p->p_owner;
    if (!x->x_has_staged) {
        pd_error(x, "ns_corpus: add with no staged expression");
        return;
    }
    const char *reason = NULL;
    int ok = do_admit(x, x->x_staged, 0.0f, 0.0f, 1, 0, &reason);
    emit_event_admit(x, ok, x->x_staged, ok ? "" : (reason ? reason : "unknown"));
    x->x_has_staged = 0;
    x->x_staged[0] = '\0';
}

static void proxy_protect(t_ns_corpus_proxy *p) {
    t_ns_corpus *x = p->p_owner;
    if (!x->x_has_staged) {
        pd_error(x, "ns_corpus: protect with no staged expression");
        return;
    }
    const char *reason = NULL;
    int ok = do_admit(x, x->x_staged, 0.0f, 0.0f, 1, 1, &reason);
    emit_event_admit(x, ok, x->x_staged, ok ? "" : (reason ? reason : "unknown"));
    x->x_has_staged = 0;
    x->x_staged[0] = '\0';
}

/* `pair` → emit `crossover <a> <b>` on the LEFT outlet so it can wire
 * straight into ns_seq_propose's right inlet for explicit-pair crossover. */
static void proxy_pair(t_ns_corpus_proxy *p) {
    t_ns_corpus *x = p->p_owner;
    if (x->x_corpus->count < 1) {
        pd_error(x, "ns_corpus: pair on empty corpus");
        return;
    }
    int idx_a = sample_one(x->x_corpus);
    int idx_b = sample_one(x->x_corpus);
    /* If only one entry, both will be the same — that's fine; crossover
     * with self just returns a copy. Caller should check size before pair
     * if it cares. */
    const char *a = x->x_corpus->entries[idx_a].expr;
    const char *b = x->x_corpus->entries[idx_b].expr;
    t_atom args[2];
    SETSYMBOL(&args[0], gensym(a));
    SETSYMBOL(&args[1], gensym(b));
    outlet_anything(x->x_out_left, gensym("crossover"), 2, args);
    emit_event_sample(x, a);
    emit_event_sample(x, b);
}

/* ======================================================================== */
/* PROXY — config / persistence                                              */
/* ======================================================================== */

static void proxy_seeds(t_ns_corpus_proxy *p, t_symbol *s) {
    t_ns_corpus *x = p->p_owner;
    if (!s || !s->s_name || s->s_name[0] == '\0') {
        pd_error(x, "ns_corpus: seeds needs a path");
        return;
    }
    FILE *f = fopen(s->s_name, "r");
    if (!f) {
        pd_error(x, "ns_corpus: cannot open seeds file: %s", s->s_name);
        return;
    }
    int n_loaded = 0, n_parse_fail = 0, n_dup = 0, n_other = 0;
    char line[NS_CORPUS_EXPR_MAX];
    while (fgets(line, sizeof(line), f)) {
        normalize_line(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        if (!parse_validates(line)) { n_parse_fail++; continue; }
        const char *reason = NULL;
        int ok = do_admit(x, line, 0.0f, 0.0f, 1, 1, &reason);
        if (ok) n_loaded++;
        else if (reason && strcmp(reason, "duplicate") == 0) n_dup++;
        else n_other++;
    }
    fclose(f);
    post("ns_corpus: loaded %d protected seeds from %s "
         "(%d parse_fail, %d duplicate, %d other)",
         n_loaded, s->s_name, n_parse_fail, n_dup, n_other);
    emit_event_size(x);
}

static int corpus_save(const ns_corpus_t *c, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    uint32_t header[4];
    header[0] = NS_CORPUS_FILE_MAGIC;
    header[1] = NS_CORPUS_FILE_VERSION;
    header[2] = (uint32_t)c->count;
    header[3] = 0;
    if (fwrite(header, sizeof(header), 1, f) != 1) { fclose(f); return 0; }
    if (c->count > 0) {
        if (fwrite(c->entries, sizeof(ns_corpus_entry_t), c->count, f)
            != (size_t)c->count) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

static int corpus_load(ns_corpus_t *c, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t header[4];
    if (fread(header, sizeof(header), 1, f) != 1) { fclose(f); return 0; }
    if (header[0] != NS_CORPUS_FILE_MAGIC) { fclose(f); return 0; }
    if (header[1] != NS_CORPUS_FILE_VERSION) { fclose(f); return 0; }
    int count = (int)header[2];
    if (count < 0) { fclose(f); return 0; }
    /* Wipe existing, preserve config. */
    c->count = 0;
    if (count == 0) { fclose(f); return 1; }
    if (!ns_corpus_reserve(c, count)) { fclose(f); return 0; }
    if (fread(c->entries, sizeof(ns_corpus_entry_t), count, f)
        != (size_t)count) { fclose(f); return 0; }
    c->count = count;
    fclose(f);
    return 1;
}

static void proxy_save(t_ns_corpus_proxy *p, t_symbol *s) {
    t_ns_corpus *x = p->p_owner;
    if (!s || !s->s_name || s->s_name[0] == '\0') {
        pd_error(x, "ns_corpus: save needs a path");
        return;
    }
    if (!corpus_save(x->x_corpus, s->s_name)) {
        pd_error(x, "ns_corpus: save failed (path: %s)", s->s_name);
    } else {
        post("ns_corpus: saved %d entries to %s", x->x_corpus->count, s->s_name);
    }
}

static void proxy_load(t_ns_corpus_proxy *p, t_symbol *s) {
    t_ns_corpus *x = p->p_owner;
    if (!s || !s->s_name || s->s_name[0] == '\0') {
        pd_error(x, "ns_corpus: load needs a path");
        return;
    }
    if (!corpus_load(x->x_corpus, s->s_name)) {
        pd_error(x, "ns_corpus: load failed (path: %s)", s->s_name);
    } else {
        post("ns_corpus: loaded %d entries from %s", x->x_corpus->count, s->s_name);
        emit_event_size(x);
    }
}

static void proxy_cap(t_ns_corpus_proxy *p, t_floatarg f) {
    int n = (int)f;
    if (n < 4) n = 4;
    p->p_owner->x_corpus->max_entries = n;
}

static void proxy_min_novelty(t_ns_corpus_proxy *p, t_floatarg f) {
    p->p_owner->x_corpus->min_novelty = (float)f;
}

static void proxy_min_quality(t_ns_corpus_proxy *p, t_floatarg f) {
    p->p_owner->x_corpus->min_quality = (float)f;
}

static void proxy_selection(t_ns_corpus_proxy *p, t_symbol *s) {
    t_ns_corpus *x = p->p_owner;
    if (!s || !s->s_name) return;
    if (strcmp(s->s_name, "uniform") == 0) {
        x->x_corpus->selection = NS_CORPUS_SEL_UNIFORM;
    } else if (strcmp(s->s_name, "weighted") == 0) {
        x->x_corpus->selection = NS_CORPUS_SEL_WEIGHTED;
    } else if (strcmp(s->s_name, "tournament") == 0) {
        x->x_corpus->selection = NS_CORPUS_SEL_TOURNAMENT;
    } else {
        pd_error(x, "ns_corpus: unknown selection '%s' (use uniform|weighted|tournament)",
                 s->s_name);
    }
}

static void proxy_seed(t_ns_corpus_proxy *p, t_floatarg f) {
    ns_rng_seed(&p->p_owner->x_corpus->rng, (uint64_t)(int64_t)f);
}

static void proxy_clear(t_ns_corpus_proxy *p) {
    ns_corpus_t *c = p->p_owner->x_corpus;
    /* Keep protected entries; drop the rest. */
    int w = 0;
    for (int r = 0; r < c->count; r++) {
        if (c->entries[r].is_protected) {
            if (w != r) c->entries[w] = c->entries[r];
            w++;
        }
    }
    c->count = w;
    emit_event_size(p->p_owner);
}

static void proxy_dump(t_ns_corpus_proxy *p) {
    t_ns_corpus *x = p->p_owner;
    ns_corpus_t *c = x->x_corpus;
    for (int i = 0; i < c->count; i++) {
        t_atom args[4];
        SETSYMBOL(&args[0], gensym(c->entries[i].expr));
        SETFLOAT(&args[1], (t_float)c->entries[i].novelty);
        SETFLOAT(&args[2], (t_float)c->entries[i].quality);
        SETFLOAT(&args[3], (t_float)c->entries[i].generation);
        outlet_anything(x->x_out_right, gensym("dump"), 4, args);
    }
}

static void proxy_size(t_ns_corpus_proxy *p) {
    emit_event_size(p->p_owner);
}

/* ======================================================================== */
/* CONSTRUCTOR / DESTRUCTOR                                                  */
/* ======================================================================== */

static void *ns_corpus_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    t_ns_corpus *x = (t_ns_corpus *)pd_new(ns_corpus_class);

    int cap = NS_CORPUS_DEFAULT_CAP;
    int ai = 0;
    x->x_name = NULL;

    /* Optional first arg: name (symbol). Optional second: capacity. */
    if (argc > ai && argv[ai].a_type == A_SYMBOL) {
        x->x_name = atom_getsymbol(&argv[ai]);
        ai++;
    }
    if (argc > ai && argv[ai].a_type == A_FLOAT) {
        int n = (int)atom_getfloat(&argv[ai]);
        if (n >= 4) cap = n;
        ai++;
    }

    if (x->x_name) {
        x->x_corpus = registry_acquire(x->x_name, cap);
    } else {
        x->x_corpus = ns_corpus_create(cap);
    }
    if (!x->x_corpus) {
        pd_error(x, "ns_corpus: failed to create corpus");
        return NULL;
    }

    x->x_staged[0] = '\0';
    x->x_has_staged = 0;

    /* Right (cold) inlet via proxy. */
    x->x_proxy.p_pd = ns_corpus_proxy_class;
    x->x_proxy.p_owner = x;
    inlet_new(&x->x_obj, &x->x_proxy.p_pd, 0, 0);

    /* Outlets. */
    x->x_out_left  = outlet_new(&x->x_obj, &s_anything);
    x->x_out_right = outlet_new(&x->x_obj, &s_anything);

    return (void *)x;
}

static void ns_corpus_free(t_ns_corpus *x) {
    if (x->x_name) {
        registry_release(x->x_corpus);
    } else {
        ns_corpus_destroy(x->x_corpus);
    }
}

/* ======================================================================== */
/* SETUP                                                                    */
/* ======================================================================== */

void ns_corpus_setup(void) {
    /* Proxy class for the right (cold) inlet. */
    ns_corpus_proxy_class = class_new(gensym("_ns_corpus_proxy"),
        0, 0, sizeof(t_ns_corpus_proxy), CLASS_PD, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_admit,
                    gensym("admit"), A_GIMME, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_add,
                    gensym("add"), 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_protect,
                    gensym("protect"), 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_pair,
                    gensym("pair"), 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_seeds,
                    gensym("seeds"), A_SYMBOL, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_save,
                    gensym("save"), A_SYMBOL, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_load,
                    gensym("load"), A_SYMBOL, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_cap,
                    gensym("cap"), A_FLOAT, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_min_novelty,
                    gensym("min_novelty"), A_FLOAT, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_min_quality,
                    gensym("min_quality"), A_FLOAT, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_selection,
                    gensym("selection"), A_SYMBOL, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_seed,
                    gensym("seed"), A_FLOAT, 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_clear,
                    gensym("clear"), 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_dump,
                    gensym("dump"), 0);
    class_addmethod(ns_corpus_proxy_class, (t_method)proxy_size,
                    gensym("size"), 0);

    /* Main class. */
    ns_corpus_class = class_new(gensym("ns_corpus"),
        (t_newmethod)ns_corpus_new,
        (t_method)ns_corpus_free,
        sizeof(t_ns_corpus),
        CLASS_DEFAULT,
        A_GIMME,
        0);

    class_addbang(ns_corpus_class, ns_corpus_bang);
    class_addanything(ns_corpus_class, ns_corpus_anything);
    class_addlist(ns_corpus_class, ns_corpus_list);
    class_addsymbol(ns_corpus_class, ns_corpus_symbol);

    post("ns_corpus %s - rolling expression corpus with quality+novelty admission",
         NS_VERSION_STRING);
}
