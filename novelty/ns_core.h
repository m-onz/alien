/*
 * ns_core.h - Shared novelty-search primitives
 *
 * Header-only library mirroring the style of alien_core.h. Provides the
 * cross-domain primitives every component in the suite depends on:
 *
 *   - memory allocation abstraction (Pd / standalone)
 *   - deterministic RNG (xoshiro256**) with Gaussian helper
 *   - distance metrics (L2, cosine, hamming)
 *   - vector archive with mean-kNN scoring + binary persistence
 *   - in-place mutation operators (Gaussian, bit-flip)
 *   - sequence → 27-dim BC features
 *
 * Domain-specific projectors live in their own headers (see ns_grid.h for
 * 2D-grid stats and ns_alien_ast.h for AST mutation/crossover) so this file
 * stays free of domain code.
 *
 * All functions are static so the header can be #included from each
 * external without a separate translation unit. Binary cost is minimal.
 */

#ifndef NS_CORE_H
#define NS_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>

/* Version */
#define NS_VERSION_MAJOR 0
#define NS_VERSION_MINOR 1
#define NS_VERSION_PATCH 0
#define NS_VERSION_STRING "0.1.0"

/* ======================================================================== */
/* MEMORY ALLOCATION                                                        */
/* ======================================================================== */

#ifdef PD
    #include "m_pd.h"
    #define NS_MALLOC(size) getbytes(size)
    #define NS_FREE(ptr, size) freebytes(ptr, size)
    #define NS_REALLOC(ptr, old_size, new_size) resizebytes(ptr, old_size, new_size)
#else
    #define NS_MALLOC(size) malloc(size)
    #define NS_FREE(ptr, size) free(ptr)
    /* Standalone realloc ignores old_size; that's fine. */
    #define NS_REALLOC(ptr, old_size, new_size) realloc(ptr, new_size)
#endif

/* ======================================================================== */
/* LIMITS                                                                   */
/* ======================================================================== */

#define NS_MAX_DIM 4096
#define NS_DEFAULT_CAPACITY 64
#define NS_NAME_MAX 64
#define NS_DEFAULT_K 15

/* On-disk format magic + version. Version 2 introduced the domain-agnostic
 * 24-dim seq-feature BC (replacing the 27-dim MIDI-biased layout). Loads
 * of v1 files fail with a clear error instead of silently misinterpreting
 * differently-shaped vectors. */
#define NS_FILE_MAGIC 0x4E534152u   /* "NSAR" little-endian */
#define NS_FILE_VERSION 2u

/* ======================================================================== */
/* DETERMINISTIC RNG (xoshiro256**)                                         */
/* ======================================================================== */

typedef struct {
    uint64_t s[4];
    /* For Box-Muller pair caching. */
    int has_spare;
    float spare;
} ns_rng_t;

static inline uint64_t ns_rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* SplitMix64 — used to expand a single seed into the 256-bit state. */
static inline uint64_t ns_rng_splitmix(uint64_t *z) {
    *z += 0x9E3779B97F4A7C15ULL;
    uint64_t r = *z;
    r = (r ^ (r >> 30)) * 0xBF58476D1CE4E5B9ULL;
    r = (r ^ (r >> 27)) * 0x94D049BB133111EBULL;
    return r ^ (r >> 31);
}

static inline void ns_rng_seed(ns_rng_t *r, uint64_t seed) {
    uint64_t z = seed ? seed : 0xDEADBEEFCAFEBABEULL;
    r->s[0] = ns_rng_splitmix(&z);
    r->s[1] = ns_rng_splitmix(&z);
    r->s[2] = ns_rng_splitmix(&z);
    r->s[3] = ns_rng_splitmix(&z);
    r->has_spare = 0;
    r->spare = 0.0f;
}

static inline uint64_t ns_rng_next(ns_rng_t *r) {
    uint64_t result = ns_rng_rotl(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = ns_rng_rotl(r->s[3], 45);
    return result;
}

/* Uniform float in [0, 1). */
static inline float ns_rng_uniform(ns_rng_t *r) {
    /* Top 24 bits → mantissa. */
    uint64_t x = ns_rng_next(r) >> 40;
    return (float)x * (1.0f / 16777216.0f);
}

/* Standard normal N(0,1) via Box-Muller. Caches the spare draw. */
static inline float ns_rng_gaussian(ns_rng_t *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    float u1, u2, mag;
    do {
        u1 = ns_rng_uniform(r);
    } while (u1 < 1e-7f);
    u2 = ns_rng_uniform(r);
    mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf((float)(2.0 * M_PI) * u2);
    float z1 = mag * sinf((float)(2.0 * M_PI) * u2);
    r->spare = z1;
    r->has_spare = 1;
    return z0;
}

/* ======================================================================== */
/* DISTANCE METRICS                                                         */
/* ======================================================================== */

typedef enum {
    NS_DIST_L2 = 0,
    NS_DIST_COSINE = 1,
    NS_DIST_HAMMING = 2,
} ns_distance_t;

static inline float ns_dist_l2(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

/* Cosine *distance* = 1 - cosine similarity. Range [0, 2]. */
static inline float ns_dist_cosine(const float *a, const float *b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < 1e-12f) return 1.0f;
    float sim = dot / denom;
    if (sim > 1.0f) sim = 1.0f;
    if (sim < -1.0f) sim = -1.0f;
    return 1.0f - sim;
}

/* Hamming distance — counts mismatches treating each element as
 * "above 0.5" (binary). Returns a normalized [0, 1] fraction. */
static inline float ns_dist_hamming(const float *a, const float *b, int dim) {
    if (dim <= 0) return 0.0f;
    int mismatches = 0;
    for (int i = 0; i < dim; i++) {
        int ba = (a[i] >= 0.5f);
        int bb = (b[i] >= 0.5f);
        if (ba != bb) mismatches++;
    }
    return (float)mismatches / (float)dim;
}

static inline float ns_distance(const float *a, const float *b, int dim, ns_distance_t metric) {
    switch (metric) {
        case NS_DIST_COSINE:  return ns_dist_cosine(a, b, dim);
        case NS_DIST_HAMMING: return ns_dist_hamming(a, b, dim);
        case NS_DIST_L2:
        default:              return ns_dist_l2(a, b, dim);
    }
}

/* ======================================================================== */
/* VECTOR ARCHIVE                                                           */
/* ======================================================================== */

typedef struct {
    int dim;                /* 0 until first add; subsequent adds must match */
    int count;              /* number of vectors stored */
    int capacity;           /* allocated capacity (in vectors) */
    float *data;            /* contiguous flat array, size = capacity * dim */
    int k;                  /* kNN k; 0 = auto = min(NS_DEFAULT_K, count/3) */
    ns_distance_t metric;   /* distance function */
    int refcount;           /* >=1 for live archives; used by named-instance registry */
} ns_archive_t;

static inline ns_archive_t *ns_archive_create(void) {
    ns_archive_t *a = (ns_archive_t *)NS_MALLOC(sizeof(ns_archive_t));
    if (!a) return NULL;
    a->dim = 0;
    a->count = 0;
    a->capacity = 0;
    a->data = NULL;
    a->k = 0;
    a->metric = NS_DIST_L2;
    a->refcount = 1;
    return a;
}

static inline void ns_archive_clear(ns_archive_t *a) {
    if (!a) return;
    if (a->data) {
        NS_FREE(a->data, sizeof(float) * a->capacity * (a->dim > 0 ? a->dim : 1));
        a->data = NULL;
    }
    a->count = 0;
    a->capacity = 0;
    a->dim = 0;
}

static inline void ns_archive_destroy(ns_archive_t *a) {
    if (!a) return;
    ns_archive_clear(a);
    NS_FREE(a, sizeof(ns_archive_t));
}

/* Grow capacity (doubling). Allocates if first-time. dim is fixed by caller. */
static inline int ns_archive_reserve(ns_archive_t *a, int new_capacity) {
    if (a->capacity >= new_capacity) return 1;
    int target = a->capacity > 0 ? a->capacity : NS_DEFAULT_CAPACITY;
    while (target < new_capacity) target *= 2;
    size_t new_bytes = sizeof(float) * target * a->dim;
    float *p;
    if (a->data) {
        size_t old_bytes = sizeof(float) * a->capacity * a->dim;
        (void)old_bytes;  /* macro expansion may not use it (standalone build) */
        p = (float *)NS_REALLOC(a->data, old_bytes, new_bytes);
    } else {
        p = (float *)NS_MALLOC(new_bytes);
    }
    if (!p) return 0;
    a->data = p;
    a->capacity = target;
    return 1;
}

/* Add a vector. First call locks the archive's dim. Subsequent calls
 * must match. Returns 1 on success, 0 on dim mismatch / OOM. */
static inline int ns_archive_add(ns_archive_t *a, const float *vec, int dim) {
    if (!a || !vec || dim <= 0 || dim > NS_MAX_DIM) return 0;
    if (a->dim == 0) {
        a->dim = dim;
    } else if (a->dim != dim) {
        return 0;  /* dim mismatch */
    }
    if (!ns_archive_reserve(a, a->count + 1)) return 0;
    memcpy(a->data + a->count * a->dim, vec, sizeof(float) * dim);
    a->count++;
    return 1;
}

/* Drop the entry at index i (swap-remove for O(1)). */
static inline void ns_archive_remove(ns_archive_t *a, int i) {
    if (!a || i < 0 || i >= a->count) return;
    int last = a->count - 1;
    if (i != last) {
        memcpy(a->data + i * a->dim,
               a->data + last * a->dim,
               sizeof(float) * a->dim);
    }
    a->count = last;
}

/* Compute mean of the k smallest distances to entries in the archive.
 * If archive is empty, returns INFINITY. */
static inline float ns_archive_score(const ns_archive_t *a, const float *vec, int dim) {
    if (!a || a->count == 0) return INFINITY;
    if (a->dim != dim) return -1.0f;  /* dim mismatch flagged with negative */

    int k = a->k > 0 ? a->k : NS_DEFAULT_K;
    if (k > a->count) k = a->count;
    if (k <= 0) k = 1;

    /* Maintain a sorted top-k of the smallest distances. Insertion-sort
     * style — fine for k up to a few dozen. */
    float top[NS_DEFAULT_K * 4];  /* cap at 60; if user sets larger, we still work */
    int max_k = (int)(sizeof(top) / sizeof(top[0]));
    if (k > max_k) k = max_k;

    /* Initialize with +inf. */
    for (int i = 0; i < k; i++) top[i] = INFINITY;

    for (int i = 0; i < a->count; i++) {
        float d = ns_distance(vec, a->data + i * a->dim, dim, a->metric);
        /* Insert into top-k if smaller than current max. */
        if (d < top[k - 1]) {
            int j = k - 1;
            while (j > 0 && top[j - 1] > d) {
                top[j] = top[j - 1];
                j--;
            }
            top[j] = d;
        }
    }

    float sum = 0.0f;
    int valid = 0;
    for (int i = 0; i < k; i++) {
        if (isfinite(top[i])) {
            sum += top[i];
            valid++;
        }
    }
    if (valid == 0) return 0.0f;
    return sum / (float)valid;
}

/* ======================================================================== */
/* PERSISTENCE                                                              */
/*                                                                          */
/* Format (little-endian):                                                  */
/*   uint32 magic        ("NSAR")                                           */
/*   uint32 version      (currently 1)                                      */
/*   uint32 dim                                                             */
/*   uint32 count                                                           */
/*   uint32 metric       (0=L2, 1=cosine, 2=hamming)                        */
/*   uint32 reserved     (zero)                                             */
/*   float32 data[count * dim]                                              */
/* ======================================================================== */

static inline int ns_archive_save(const ns_archive_t *a, const char *path) {
    if (!a || !path) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    uint32_t header[6];
    header[0] = NS_FILE_MAGIC;
    header[1] = NS_FILE_VERSION;
    header[2] = (uint32_t)a->dim;
    header[3] = (uint32_t)a->count;
    header[4] = (uint32_t)a->metric;
    header[5] = 0;
    if (fwrite(header, sizeof(header), 1, f) != 1) { fclose(f); return 0; }
    if (a->count > 0 && a->dim > 0) {
        size_t n = (size_t)a->count * (size_t)a->dim;
        if (fwrite(a->data, sizeof(float), n, f) != n) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

/* Replaces the archive's contents (preserves refcount). */
static inline int ns_archive_load(ns_archive_t *a, const char *path) {
    if (!a || !path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t header[6];
    if (fread(header, sizeof(header), 1, f) != 1) { fclose(f); return 0; }
    if (header[0] != NS_FILE_MAGIC) { fclose(f); return 0; }
    if (header[1] != NS_FILE_VERSION) { fclose(f); return 0; }
    int dim = (int)header[2];
    int count = (int)header[3];
    if (dim < 1 || dim > NS_MAX_DIM || count < 0) { fclose(f); return 0; }

    /* Wipe existing data while preserving everything else (k, refcount). */
    ns_archive_clear(a);
    a->metric = (ns_distance_t)header[4];
    if (count == 0) { fclose(f); return 1; }

    /* Allocate exactly count slots. */
    a->dim = dim;
    if (!ns_archive_reserve(a, count)) { fclose(f); return 0; }

    size_t n = (size_t)count * (size_t)dim;
    if (fread(a->data, sizeof(float), n, f) != n) { fclose(f); return 0; }
    a->count = count;
    fclose(f);
    return 1;
}

/* ======================================================================== */
/* MUTATION OPERATORS                                                       */
/* ======================================================================== */

/* Add N(0, sigma) to each element in-place. */
static inline void ns_mutate_gaussian(float *vec, int dim, float sigma, ns_rng_t *r) {
    if (!vec || dim <= 0 || sigma <= 0.0f || !r) return;
    for (int i = 0; i < dim; i++) {
        vec[i] += ns_rng_gaussian(r) * sigma;
    }
}

/* For each element, with probability `prob`, flip its binary state
 * (treating >=0.5 as 1, <0.5 as 0). */
static inline void ns_mutate_bitflip(float *vec, int dim, float prob, ns_rng_t *r) {
    if (!vec || dim <= 0 || prob <= 0.0f || !r) return;
    for (int i = 0; i < dim; i++) {
        if (ns_rng_uniform(r) < prob) {
            vec[i] = (vec[i] >= 0.5f) ? 0.0f : 1.0f;
        }
    }
}

static inline void ns_clip(float *vec, int dim, float lo, float hi) {
    if (!vec || dim <= 0) return;
    for (int i = 0; i < dim; i++) {
        if (vec[i] < lo) vec[i] = lo;
        else if (vec[i] > hi) vec[i] = hi;
    }
}

/* 2D grid statistics live in ns_grid.h (domain-specific projector). Pulled
 * in here so callers that #include only ns_core.h still resolve ns_grid_stats. */
#include "ns_grid.h"

/* ======================================================================== */
/* SEQUENCE FEATURES — domain-agnostic 24-dim BC                            */
/*                                                                          */
/* Treats seq[] as an opaque list of integers (whatever they mean — MIDI    */
/* notes, sample-folder indices, parameter values, wavetable slots) plus    */
/* the sentinel NS_REST for silences. The BC vector is invariant to affine */
/* scaling/translation of the value space: scaling all numbers by 10×      */
/* leaves the BC unchanged. This is what enables the same novelty search   */
/* across musical, parametric, and visual domains without retuning.         */
/*                                                                          */
/* Output layout (24 floats, all in [0, 1]):                                */
/*    0   length_norm       len(seq) / NS_SEQ_MAX_OUTPUT_LEN, capped        */
/*    1   rest_ratio        fraction of rests                               */
/*    2   unique_ratio      |distinct notes| / |notes|                      */
/*    3   mean_rank         mean of normalised ranks (rank/(distinct-1))    */
/*    4   range_norm        saturating function of distinct count           */
/*    5   rank_std          std of normalised ranks                         */
/*    6   ascend            fraction of intervals > 0                       */
/*    7   descend           fraction of intervals < 0                       */
/*    8   repeat            fraction of intervals = 0                       */
/*    9   rhythm_ent        run-length Shannon entropy of note/rest mask    */
/*    10  ac1               lag-1 mask autocorr, mapped to [0,1]            */
/*    11  ac2               lag-2 mask autocorr, mapped to [0,1]            */
/*    12-19  step_hist[8]   |step|/observed_range, 8 bins:                  */
/*                          [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]    */
/*    20-23  value_hist[4]  (v - min) / range, 4 equal bins on [0,1]        */
/* ======================================================================== */

#define NS_REST INT_MIN                 /* rest sentinel */
#define NS_SEQ_MAX_LEN 1024
#define NS_SEQ_MAX_OUTPUT_LEN 96        /* normalisation constant for length */
#define NS_SEQ_FEATURE_DIM 24
#define NS_STEP_BIN_COUNT 8
#define NS_VALUE_BIN_COUNT 4

/* Upper edges of step-magnitude bins (relative to observed range).
 * A normalised |step| falling at-or-below an edge lands in that bin.
 * The 8th bin's effective upper edge is 1.0 (post-clamp). */
static const float NS_STEP_BIN_EDGES[NS_STEP_BIN_COUNT] = {
    0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f
};

static inline int ns_seq_is_rest(int v) {
    return v == NS_REST;
}

static inline void ns_seq_features(const int *seq, int n_in, float *out) {
    /* Always zero the output first so partial returns are safe. */
    for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) out[i] = 0.0f;

    if (!seq || n_in <= 0 || !out) return;

    int n = (n_in > NS_SEQ_MAX_LEN) ? NS_SEQ_MAX_LEN : n_in;

    /* Stack buffers — small but plenty for music sequences. */
    int notes_buf[NS_SEQ_MAX_LEN];
    int intervals_buf[NS_SEQ_MAX_LEN];
    int runs_buf[NS_SEQ_MAX_LEN];
    int sorted_distinct[NS_SEQ_MAX_LEN];

    /* 0. length_norm */
    float length_norm = (float)n / (float)NS_SEQ_MAX_OUTPUT_LEN;
    if (length_norm > 1.0f) length_norm = 1.0f;

    /* 1. rest_ratio */
    int rest_count = 0;
    for (int i = 0; i < n; i++) if (ns_seq_is_rest(seq[i])) rest_count++;
    float rest_ratio = (float)rest_count / (float)n;

    /* Collect notes (non-rest values). */
    int nc = 0;
    for (int i = 0; i < n; i++) {
        if (!ns_seq_is_rest(seq[i])) notes_buf[nc++] = seq[i];
    }

    /* Build sorted-distinct list for rank lookups. O(n²) but n is bounded. */
    int distinct_count = 0;
    for (int i = 0; i < nc; i++) {
        int dup = 0;
        for (int j = 0; j < distinct_count; j++) {
            if (sorted_distinct[j] == notes_buf[i]) { dup = 1; break; }
        }
        if (!dup) sorted_distinct[distinct_count++] = notes_buf[i];
    }
    /* Insertion sort to obtain rank order. */
    for (int i = 1; i < distinct_count; i++) {
        int key = sorted_distinct[i];
        int j = i - 1;
        while (j >= 0 && sorted_distinct[j] > key) {
            sorted_distinct[j + 1] = sorted_distinct[j];
            j--;
        }
        sorted_distinct[j + 1] = key;
    }

    /* 2. unique_ratio */
    float unique_ratio = (nc > 0) ? (float)distinct_count / (float)nc : 0.0f;

    /* Observed range for step / value bin normalisation. */
    int nmin = 0, nmax = 0, range = 0;
    if (distinct_count > 0) {
        nmin = sorted_distinct[0];
        nmax = sorted_distinct[distinct_count - 1];
        range = nmax - nmin;
    }

    /* 3-5. Rank-based distribution stats — invariant to scale/translation. */
    float mean_rank = 0.0f, range_norm = 0.0f, rank_std = 0.0f;
    if (nc > 0) {
        float rank_sum = 0.0f, rank_sum_sq = 0.0f;
        for (int i = 0; i < nc; i++) {
            int r_idx = 0;
            for (int j = 0; j < distinct_count; j++) {
                if (sorted_distinct[j] == notes_buf[i]) { r_idx = j; break; }
            }
            float rn = (distinct_count > 1)
                ? (float)r_idx / (float)(distinct_count - 1)
                : 0.5f;  /* single distinct value → middle */
            rank_sum += rn;
            rank_sum_sq += rn * rn;
        }
        mean_rank = rank_sum / (float)nc;
        float var_rank = rank_sum_sq / (float)nc - mean_rank * mean_rank;
        if (var_rank < 0.0f) var_rank = 0.0f;
        rank_std = sqrtf(var_rank);
        /* range_norm: 0 if all-same, saturating to 1 with more distinct
         * values. Captures "how much spread does this sequence cover" in
         * a domain-agnostic way. */
        range_norm = 1.0f - 1.0f / (float)(distinct_count > 0 ? distinct_count : 1);
    }

    /* 6-8 + step inputs: walk the sequence, skip rests, record intervals. */
    int interval_count = 0;
    int ascend_count = 0, descend_count = 0, repeat_count = 0;
    int has_prev = 0, prev_note = 0;
    for (int i = 0; i < n; i++) {
        if (ns_seq_is_rest(seq[i])) continue;
        if (has_prev) {
            int d = seq[i] - prev_note;
            if (d > 0) ascend_count++;
            else if (d < 0) descend_count++;
            else repeat_count++;
            if (interval_count < NS_SEQ_MAX_LEN) {
                intervals_buf[interval_count] = d;
            }
            interval_count++;
        }
        prev_note = seq[i];
        has_prev = 1;
    }
    float ascend = 0.0f, descend = 0.0f, repeat = 0.0f;
    if (interval_count > 0) {
        ascend  = (float)ascend_count  / (float)interval_count;
        descend = (float)descend_count / (float)interval_count;
        repeat  = (float)repeat_count  / (float)interval_count;
    }

    /* 9. rhythm_ent — Shannon entropy of run lengths over note/rest mask. */
    float rhythm_ent = 0.0f;
    {
        int run_count = 0;
        int cur = ns_seq_is_rest(seq[0]) ? 0 : 1;
        int count = 1;
        for (int i = 1; i < n; i++) {
            int x = ns_seq_is_rest(seq[i]) ? 0 : 1;
            if (x == cur) {
                count++;
            } else {
                if (run_count < NS_SEQ_MAX_LEN) runs_buf[run_count++] = count;
                cur = x;
                count = 1;
            }
        }
        if (run_count < NS_SEQ_MAX_LEN) runs_buf[run_count++] = count;

        if (run_count > 0) {
            int total = 0;
            for (int i = 0; i < run_count; i++) total += runs_buf[i];
            float ent = 0.0f;
            for (int i = 0; i < run_count; i++) {
                if (runs_buf[i] > 0) {
                    float p = (float)runs_buf[i] / (float)total;
                    ent -= p * (logf(p) / logf(2.0f));
                }
            }
            int dn = (run_count > 2) ? run_count : 2;
            float denom = logf((float)dn) / logf(2.0f);
            rhythm_ent = ent / denom;
        }
    }

    /* 10-11. lag-1 and lag-2 mask autocorrelation, remapped to [0, 1]. */
    float ac1 = 0.5f, ac2 = 0.5f;
    {
        float mask_mean = (float)(n - rest_count) / (float)n;
        float den = 0.0f;
        for (int i = 0; i < n; i++) {
            float x = ns_seq_is_rest(seq[i]) ? 0.0f : 1.0f;
            den += (x - mask_mean) * (x - mask_mean);
        }
        if (den < 1e-9f) den = 1.0f;

        if (n > 1) {
            float num = 0.0f;
            for (int i = 0; i < n - 1; i++) {
                float xi = ns_seq_is_rest(seq[i])     ? 0.0f : 1.0f;
                float xj = ns_seq_is_rest(seq[i + 1]) ? 0.0f : 1.0f;
                num += (xi - mask_mean) * (xj - mask_mean);
            }
            float a = num / den;
            if (a > 1.0f) a = 1.0f;
            if (a < -1.0f) a = -1.0f;
            ac1 = (a + 1.0f) * 0.5f;
        }
        if (n > 2) {
            float num = 0.0f;
            for (int i = 0; i < n - 2; i++) {
                float xi = ns_seq_is_rest(seq[i])     ? 0.0f : 1.0f;
                float xj = ns_seq_is_rest(seq[i + 2]) ? 0.0f : 1.0f;
                num += (xi - mask_mean) * (xj - mask_mean);
            }
            float a = num / den;
            if (a > 1.0f) a = 1.0f;
            if (a < -1.0f) a = -1.0f;
            ac2 = (a + 1.0f) * 0.5f;
        }
    }

    /* 12-19. step magnitude histogram — |step| / observed_range, 8 bins.
     * Domain-agnostic: scaling all values by a constant leaves this
     * unchanged because both step and range scale together. */
    float step_hist[NS_STEP_BIN_COUNT];
    for (int i = 0; i < NS_STEP_BIN_COUNT; i++) step_hist[i] = 0.0f;
    if (range > 0 && interval_count > 0) {
        float frange = (float)range;
        int int_total = (interval_count < NS_SEQ_MAX_LEN)
            ? interval_count : NS_SEQ_MAX_LEN;
        for (int i = 0; i < int_total; i++) {
            int d = intervals_buf[i];
            float ad = (float)((d < 0) ? -d : d);
            float rel = ad / frange;
            if (rel > 1.0f) rel = 1.0f;
            int bin = NS_STEP_BIN_COUNT - 1;
            for (int b = 0; b < NS_STEP_BIN_COUNT; b++) {
                if (rel <= NS_STEP_BIN_EDGES[b]) { bin = b; break; }
            }
            step_hist[bin] += 1.0f;
        }
        float total = 0.0f;
        for (int i = 0; i < NS_STEP_BIN_COUNT; i++) total += step_hist[i];
        if (total > 0.0f) {
            for (int i = 0; i < NS_STEP_BIN_COUNT; i++) step_hist[i] /= total;
        }
    }
    /* When range == 0 (all same value or single note) every step is 0,
     * which lands in the smallest-step bin. Encode that to keep the
     * representation distinguishable from the empty-interval case. */
    else if (interval_count > 0) {
        step_hist[0] = 1.0f;
    }

    /* 20-23. value-shape histogram — (v - min) / range, 4 equal bins.
     * Captures the distribution shape (uniform / clustered low / etc.)
     * regardless of absolute value magnitude. */
    float value_hist[NS_VALUE_BIN_COUNT];
    for (int i = 0; i < NS_VALUE_BIN_COUNT; i++) value_hist[i] = 0.0f;
    if (nc > 0 && range > 0) {
        for (int i = 0; i < nc; i++) {
            float rel = (float)(notes_buf[i] - nmin) / (float)range;
            if (rel < 0.0f) rel = 0.0f;
            if (rel > 1.0f) rel = 1.0f;
            int bin = (int)(rel * (float)NS_VALUE_BIN_COUNT);
            if (bin >= NS_VALUE_BIN_COUNT) bin = NS_VALUE_BIN_COUNT - 1;
            value_hist[bin] += 1.0f;
        }
        for (int i = 0; i < NS_VALUE_BIN_COUNT; i++) value_hist[i] /= (float)nc;
    } else if (nc > 0) {
        /* All notes identical → concentrate in bin 0 (the "low end" of
         * the empty range). Distinct from the no-notes case. */
        value_hist[0] = 1.0f;
    }

    /* Pack output. */
    out[0]  = length_norm;
    out[1]  = rest_ratio;
    out[2]  = unique_ratio;
    out[3]  = mean_rank;
    out[4]  = range_norm;
    out[5]  = rank_std;
    out[6]  = ascend;
    out[7]  = descend;
    out[8]  = repeat;
    out[9]  = rhythm_ent;
    out[10] = ac1;
    out[11] = ac2;
    for (int i = 0; i < NS_STEP_BIN_COUNT; i++)  out[12 + i] = step_hist[i];
    for (int i = 0; i < NS_VALUE_BIN_COUNT; i++) out[20 + i] = value_hist[i];
}

#endif /* NS_CORE_H */
