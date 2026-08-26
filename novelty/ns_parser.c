/*
 * ns_parser - Standalone CLI test harness for the ns_core library
 *
 * Validates RNG, distance metrics, archive (add/score/save/load),
 * mutation, and grid statistics without requiring Pure Data.
 *
 * Usage:
 *     ./ns_parser --test
 *     ./ns_parser --grid 4 4 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0
 *     ./ns_parser --score 3 1 0 0 -- 0 1 0 -- 0 0 1
 */

#include "ns_core.h"
#include "ns_alien_ast.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================== */
/* TEST INFRASTRUCTURE                                                      */
/* ======================================================================== */

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define EXPECT(cond, msg) do {                                              \
    g_tests_run++;                                                          \
    if (!(cond)) {                                                          \
        g_tests_failed++;                                                   \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);    \
    }                                                                       \
} while (0)

#define EXPECT_NEAR(actual, expected, tol, msg) do {                        \
    g_tests_run++;                                                          \
    double _a = (double)(actual), _e = (double)(expected);                  \
    if (fabs(_a - _e) > (double)(tol)) {                                    \
        g_tests_failed++;                                                   \
        fprintf(stderr, "  FAIL [%s:%d] %s — got %.6f, expected %.6f\n",    \
                __FILE__, __LINE__, msg, _a, _e);                           \
    }                                                                       \
} while (0)

static void section(const char *name) {
    printf("\n[%s]\n", name);
}

/* ======================================================================== */
/* RNG TESTS                                                                */
/* ======================================================================== */

static void test_rng(void) {
    section("rng");

    ns_rng_t a, b;
    ns_rng_seed(&a, 42);
    ns_rng_seed(&b, 42);
    /* Same seed → same sequence. */
    for (int i = 0; i < 100; i++) {
        EXPECT(ns_rng_next(&a) == ns_rng_next(&b), "rng deterministic by seed");
    }

    ns_rng_seed(&a, 42);
    ns_rng_seed(&b, 43);
    int diffs = 0;
    for (int i = 0; i < 100; i++) {
        if (ns_rng_next(&a) != ns_rng_next(&b)) diffs++;
    }
    EXPECT(diffs > 90, "different seeds produce different sequences");

    /* Uniform within [0, 1). */
    ns_rng_seed(&a, 12345);
    int in_range = 0;
    double sum = 0.0;
    for (int i = 0; i < 10000; i++) {
        float u = ns_rng_uniform(&a);
        if (u >= 0.0f && u < 1.0f) in_range++;
        sum += u;
    }
    EXPECT(in_range == 10000, "uniform stays in [0,1)");
    EXPECT_NEAR(sum / 10000.0, 0.5, 0.02, "uniform mean ≈ 0.5");

    /* Gaussian: mean ≈ 0, var ≈ 1 over many draws. */
    ns_rng_seed(&a, 999);
    double gsum = 0.0, gsumsq = 0.0;
    int N = 20000;
    for (int i = 0; i < N; i++) {
        float g = ns_rng_gaussian(&a);
        gsum += g;
        gsumsq += g * g;
    }
    EXPECT_NEAR(gsum / N, 0.0, 0.05, "gaussian mean ≈ 0");
    EXPECT_NEAR(gsumsq / N, 1.0, 0.05, "gaussian variance ≈ 1");
}

/* ======================================================================== */
/* DISTANCE TESTS                                                           */
/* ======================================================================== */

static void test_distance(void) {
    section("distance");

    float a[3] = {1, 0, 0};
    float b[3] = {0, 1, 0};
    float c[3] = {1, 0, 0};

    EXPECT_NEAR(ns_dist_l2(a, c, 3), 0.0, 1e-6, "l2 self-distance is 0");
    EXPECT_NEAR(ns_dist_l2(a, b, 3), sqrtf(2.0f), 1e-6, "l2 of orthogonal unit vectors = sqrt(2)");

    EXPECT_NEAR(ns_dist_cosine(a, c, 3), 0.0, 1e-6, "cosine self = 0");
    EXPECT_NEAR(ns_dist_cosine(a, b, 3), 1.0, 1e-6, "cosine of orthogonal = 1");

    float d[4] = {1, 0, 1, 0};
    float e[4] = {1, 1, 0, 0};
    EXPECT_NEAR(ns_dist_hamming(d, e, 4), 0.5, 1e-6, "hamming 4-bit, 2 mismatches = 0.5");

    float f[4] = {0, 0, 0, 0};
    EXPECT_NEAR(ns_dist_hamming(d, f, 4), 0.5, 1e-6, "hamming sparse vs zero");
}

/* ======================================================================== */
/* ARCHIVE TESTS                                                            */
/* ======================================================================== */

static void test_archive_basic(void) {
    section("archive basic");

    ns_archive_t *a = ns_archive_create();
    EXPECT(a != NULL, "create");
    EXPECT(a->count == 0, "starts empty");
    EXPECT(a->dim == 0, "dim is unset until first add");

    /* Empty archive returns +inf novelty. */
    float v[3] = {1, 2, 3};
    float score = ns_archive_score(a, v, 3);
    EXPECT(isinf(score), "empty archive → +inf");

    /* Add and re-query: distance to self should be 0. */
    EXPECT(ns_archive_add(a, v, 3) == 1, "add succeeds");
    EXPECT(a->dim == 3, "dim is set after first add");
    EXPECT(a->count == 1, "count = 1 after one add");
    EXPECT_NEAR(ns_archive_score(a, v, 3), 0.0, 1e-6, "self-score = 0");

    /* Dim mismatch is rejected. */
    float w[2] = {1, 2};
    EXPECT(ns_archive_add(a, w, 2) == 0, "dim mismatch rejected");

    /* Add several distinct vectors, query a far-away one. */
    float v1[3] = {0, 0, 0};
    float v2[3] = {1, 0, 0};
    float v3[3] = {0, 1, 0};
    float v4[3] = {0, 0, 1};
    ns_archive_clear(a);
    ns_archive_add(a, v1, 3);
    ns_archive_add(a, v2, 3);
    ns_archive_add(a, v3, 3);
    ns_archive_add(a, v4, 3);
    EXPECT(a->count == 4, "4 entries after 4 adds");

    float far[3] = {10, 10, 10};
    float fs = ns_archive_score(a, far, 3);
    EXPECT(fs > 5.0f, "far-away point has high novelty");

    ns_archive_destroy(a);
}

static void test_archive_grow(void) {
    section("archive grow");

    ns_archive_t *a = ns_archive_create();
    /* Add many vectors; check capacity growth and integrity. */
    for (int i = 0; i < 1000; i++) {
        float vec[4] = {(float)i, (float)(i % 7), (float)(i * 0.5), (float)(i % 13)};
        ns_archive_add(a, vec, 4);
    }
    EXPECT(a->count == 1000, "1000 entries");
    EXPECT(a->capacity >= 1000, "capacity grew");

    /* Verify a stored vector is intact. */
    float *v500 = a->data + 500 * 4;
    EXPECT(v500[0] == 500.0f, "vector 500 intact");
    EXPECT(v500[1] == (float)(500 % 7), "vector 500 col 1 intact");

    ns_archive_destroy(a);
}

static void test_archive_persistence(void) {
    section("archive save/load");

    const char *path = "/tmp/ns_test_archive.bin";
    ns_archive_t *a = ns_archive_create();
    a->metric = NS_DIST_COSINE;
    a->k = 7;
    for (int i = 0; i < 50; i++) {
        float vec[5] = {(float)i, (float)(i + 1), (float)(i * 2),
                        (float)(i % 3), (float)(50 - i)};
        ns_archive_add(a, vec, 5);
    }
    EXPECT(ns_archive_save(a, path) == 1, "save succeeds");

    ns_archive_t *b = ns_archive_create();
    EXPECT(ns_archive_load(b, path) == 1, "load succeeds");
    EXPECT(b->count == 50, "loaded count matches");
    EXPECT(b->dim == 5, "loaded dim matches");
    EXPECT(b->metric == NS_DIST_COSINE, "loaded metric matches");

    /* Spot-check a few entries. */
    for (int i = 0; i < 50; i += 7) {
        float *orig = a->data + i * 5;
        float *load = b->data + i * 5;
        for (int j = 0; j < 5; j++) {
            EXPECT(orig[j] == load[j], "byte-identical entry after round trip");
        }
    }

    /* Bad path. */
    EXPECT(ns_archive_load(b, "/tmp/this_does_not_exist_42") == 0, "missing file rejected");

    /* Bad magic — write a corrupt file. */
    FILE *f = fopen(path, "wb");
    if (f) {
        uint32_t header[6] = {0xDEADBEEF, 1, 5, 0, 0, 0};
        fwrite(header, sizeof(header), 1, f);
        fclose(f);
        EXPECT(ns_archive_load(b, path) == 0, "bad magic rejected");
    }

    ns_archive_destroy(a);
    ns_archive_destroy(b);
}

/* ======================================================================== */
/* MUTATION TESTS                                                           */
/* ======================================================================== */

static void test_mutation(void) {
    section("mutation");

    ns_rng_t r;
    ns_rng_seed(&r, 12345);

    /* Gaussian preserves length and shifts mean within tolerance. */
    float v[100];
    for (int i = 0; i < 100; i++) v[i] = 0.5f;
    float orig[100];
    memcpy(orig, v, sizeof(orig));

    ns_mutate_gaussian(v, 100, 0.1f, &r);
    int changed = 0;
    for (int i = 0; i < 100; i++) if (v[i] != orig[i]) changed++;
    EXPECT(changed > 80, "gaussian changes most cells");

    double diffsum = 0.0;
    for (int i = 0; i < 100; i++) diffsum += (v[i] - orig[i]);
    EXPECT_NEAR(diffsum / 100.0, 0.0, 0.05, "gaussian shift roughly zero-mean");

    /* Bit-flip: with prob = 1.0, every cell should flip. */
    float b[8] = {0, 1, 0, 1, 1, 1, 0, 0};
    float bcopy[8];
    memcpy(bcopy, b, sizeof(b));
    ns_mutate_bitflip(b, 8, 1.0f, &r);
    for (int i = 0; i < 8; i++) {
        EXPECT((b[i] >= 0.5f) != (bcopy[i] >= 0.5f), "p=1 flips every cell");
    }

    /* Bit-flip with prob = 0 changes nothing. */
    float c[8] = {0, 1, 0, 1, 1, 1, 0, 0};
    float ccopy[8];
    memcpy(ccopy, c, sizeof(c));
    ns_mutate_bitflip(c, 8, 0.0f, &r);
    for (int i = 0; i < 8; i++) {
        EXPECT(c[i] == ccopy[i], "p=0 changes nothing");
    }

    /* Clip. */
    float d[5] = {-1, 0, 0.5, 1, 2};
    ns_clip(d, 5, 0.0f, 1.0f);
    EXPECT(d[0] == 0.0f, "clip floor");
    EXPECT(d[3] == 1.0f, "clip ceiling");
    EXPECT(d[2] == 0.5f, "in-range untouched");
}

/* ======================================================================== */
/* GRID STATS TESTS                                                         */
/* ======================================================================== */

static void test_grid_stats(void) {
    section("grid stats");

    /* 4×4 all zeros. */
    float zero[16] = {0};
    float out[6];
    ns_grid_stats(zero, 4, 4, out);
    EXPECT_NEAR(out[0], 0.0, 1e-6, "all-zero density");
    EXPECT_NEAR(out[2], 1.0, 1e-6, "all-zero is symmetric horizontally");
    EXPECT_NEAR(out[3], 1.0, 1e-6, "all-zero is symmetric vertically");
    EXPECT_NEAR(out[4], 0.0, 1e-6, "all-zero has 0 components");

    /* 4×4 all ones. */
    float ones[16];
    for (int i = 0; i < 16; i++) ones[i] = 1.0f;
    ns_grid_stats(ones, 4, 4, out);
    EXPECT_NEAR(out[0], 1.0, 1e-6, "all-ones density");
    EXPECT_NEAR(out[1], 0.0, 1e-6, "all-ones spatial entropy = 0 (one bin)");
    EXPECT_NEAR(out[2], 1.0, 1e-6, "all-ones symmetric horizontally");
    EXPECT_NEAR(out[3], 1.0, 1e-6, "all-ones symmetric vertically");
    /* one component covering 16 cells → component_count = 1, mean_size = 1.0 */
    EXPECT_NEAR(out[5], 1.0, 1e-6, "all-ones single component, full size");

    /* 4×4 checkerboard: each cell isolated → 8 components in even checkerboard. */
    float check[16] = {
        0,1,0,1,
        1,0,1,0,
        0,1,0,1,
        1,0,1,0,
    };
    ns_grid_stats(check, 4, 4, out);
    EXPECT_NEAR(out[0], 0.5, 1e-6, "checkerboard density 0.5");
    /* 8 isolated alive cells → 8 components, each size 1 */
    /* normalized: 8 / (16/2) = 1.0 */
    EXPECT_NEAR(out[4], 1.0, 1e-6, "checkerboard maxes out components");
    EXPECT_NEAR(out[5], 1.0/16.0, 1e-6, "checkerboard mean component size = 1/16");

    /* Horizontal stripe: top half on, bottom half off → not h-symmetric. */
    float stripe_h[16] = {
        1,1,1,1,
        1,1,1,1,
        0,0,0,0,
        0,0,0,0,
    };
    ns_grid_stats(stripe_h, 4, 4, out);
    EXPECT_NEAR(out[0], 0.5, 1e-6, "h-stripe density");
    EXPECT_NEAR(out[2], 0.0, 1e-6, "h-stripe is anti-symmetric horizontally");
    EXPECT_NEAR(out[3], 1.0, 1e-6, "h-stripe is symmetric vertically");

    /* Vertical stripe. */
    float stripe_v[16] = {
        1,1,0,0,
        1,1,0,0,
        1,1,0,0,
        1,1,0,0,
    };
    ns_grid_stats(stripe_v, 4, 4, out);
    EXPECT_NEAR(out[2], 1.0, 1e-6, "v-stripe is symmetric horizontally");
    EXPECT_NEAR(out[3], 0.0, 1e-6, "v-stripe is anti-symmetric vertically");
    EXPECT_NEAR(out[4], 1.0/8.0, 1e-6, "v-stripe has 1 component, normalized 1/8");

    /* 16×16 random sanity: just ensure no crash and output is bounded. */
    float big[256];
    ns_rng_t rng; ns_rng_seed(&rng, 7);
    for (int i = 0; i < 256; i++) big[i] = ns_rng_uniform(&rng);
    ns_grid_stats(big, 16, 16, out);
    for (int i = 0; i < 6; i++) {
        EXPECT(out[i] >= 0.0f && out[i] <= 1.0f, "16x16 outputs in [0,1]");
    }
}

/* ======================================================================== */
/* SEQUENCE FEATURES TESTS                                                  */
/* ======================================================================== */

static void test_seq_features(void) {
    section("seq features");

    float feat[NS_SEQ_FEATURE_DIM];

    /* All-rest sequence: most features collapse to defaults. */
    int all_rest[8] = {NS_REST, NS_REST, NS_REST, NS_REST,
                       NS_REST, NS_REST, NS_REST, NS_REST};
    ns_seq_features(all_rest, 8, feat);
    EXPECT_NEAR(feat[1], 1.0, 1e-6, "all-rest → rest_ratio = 1");
    EXPECT_NEAR(feat[2], 0.0, 1e-6, "all-rest → unique_ratio = 0");
    EXPECT_NEAR(feat[3], 0.0, 1e-6, "all-rest → mean_rank = 0");
    EXPECT_NEAR(feat[4], 0.0, 1e-6, "all-rest → range_norm = 0");
    EXPECT_NEAR(feat[5], 0.0, 1e-6, "all-rest → rank_std = 0");
    EXPECT_NEAR(feat[6], 0.0, 1e-6, "all-rest → ascend = 0");
    EXPECT_NEAR(feat[10], 0.5, 1e-6, "all-rest → ac1 maps to 0.5 (0 corr)");

    /* Single repeated note: one distinct value, all-zero steps,
     * 100% repeat intervals. range_norm saturates as 1 - 1/distinct. */
    int monotone[6] = {60, 60, 60, 60, 60, 60};
    ns_seq_features(monotone, 6, feat);
    EXPECT_NEAR(feat[1], 0.0, 1e-6, "monotone → rest_ratio = 0");
    EXPECT_NEAR(feat[2], 1.0/6.0, 1e-6, "monotone → unique_ratio = 1/6");
    EXPECT_NEAR(feat[3], 0.5, 1e-6, "monotone → mean_rank = 0.5 (single distinct)");
    EXPECT_NEAR(feat[4], 0.0, 1e-6, "monotone → range_norm = 0 (1 - 1/1)");
    EXPECT_NEAR(feat[5], 0.0, 1e-6, "monotone → rank_std = 0");
    EXPECT_NEAR(feat[8], 1.0, 1e-6, "monotone → repeat = 1");
    EXPECT_NEAR(feat[6], 0.0, 1e-6, "monotone → ascend = 0");
    /* All steps are 0 → land in smallest-magnitude bin (step_hist[0]). */
    EXPECT_NEAR(feat[12], 1.0, 1e-6, "monotone → all steps in smallest bin");
    /* All values identical → value_hist concentrates in bin 0. */
    EXPECT_NEAR(feat[20], 1.0, 1e-6, "monotone → value_hist in low bin");

    /* Ascending sequence in arbitrary value space (could be MIDI, sample
     * indices, params — agnostic). 8 distinct values. */
    int scale[8] = {60, 62, 64, 65, 67, 69, 71, 72};
    ns_seq_features(scale, 8, feat);
    EXPECT_NEAR(feat[1], 0.0, 1e-6, "scale → rest_ratio = 0");
    EXPECT_NEAR(feat[2], 1.0, 1e-6, "scale → all unique");
    EXPECT_NEAR(feat[6], 1.0, 1e-6, "scale → ascend = 1");
    EXPECT_NEAR(feat[7], 0.0, 1e-6, "scale → descend = 0");
    EXPECT_NEAR(feat[8], 0.0, 1e-6, "scale → repeat = 0");
    EXPECT(feat[4] > 0.5f, "scale → range_norm reflects 8 distinct values");
    EXPECT(feat[5] > 0.0f, "scale → rank_std nonzero");

    /* Domain-agnosticism: the same shape scaled by 100× should produce
     * an identical BC vector. Critical invariant. */
    int scale_scaled[8] = {6000, 6200, 6400, 6500, 6700, 6900, 7100, 7200};
    float feat2[NS_SEQ_FEATURE_DIM];
    ns_seq_features(scale_scaled, 8, feat2);
    int identical = 1;
    for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) {
        if (feat[i] != feat2[i]) { identical = 0; break; }
    }
    EXPECT(identical, "scale-invariance: BC unchanged under value-space scaling");

    /* Same shape, different absolute origin (translation). Also invariant. */
    int scale_translated[8] = {1060, 1062, 1064, 1065, 1067, 1069, 1071, 1072};
    ns_seq_features(scale_translated, 8, feat2);
    identical = 1;
    for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) {
        if (feat[i] != feat2[i]) { identical = 0; break; }
    }
    EXPECT(identical, "translation-invariance: BC unchanged under value-space shift");

    /* Note-rest alternating pattern: rhythm/autocorr unchanged from old version.
     * Mask = [1,0,1,0,1,0,1,0]; lag-1 ≈ -7/8 → 0.0625, lag-2 ≈ +6/8 → 0.875. */
    int alt[8] = {60, NS_REST, 64, NS_REST, 67, NS_REST, 60, NS_REST};
    ns_seq_features(alt, 8, feat);
    EXPECT_NEAR(feat[1], 0.5, 1e-6, "alt → rest_ratio = 0.5");
    EXPECT_NEAR(feat[10], 0.0625, 0.01, "alt → lag-1 ac ≈ 0.0625 (strong anti-corr)");
    EXPECT_NEAR(feat[11], 0.875, 0.01, "alt → lag-2 ac ≈ 0.875 (period-2)");

    /* Output bounds: every feature in [0, 1]. */
    int mixed[10] = {60, 62, NS_REST, 64, 65, 64, NS_REST, 67, 60, 72};
    ns_seq_features(mixed, 10, feat);
    for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) {
        EXPECT(feat[i] >= 0.0f && feat[i] <= 1.0f + 1e-5f,
               "seq feat in [0,1]");
    }

    /* Empty input: all zeros. */
    ns_seq_features(NULL, 0, feat);
    int all_zero = 1;
    for (int i = 0; i < NS_SEQ_FEATURE_DIM; i++) {
        if (feat[i] != 0.0f) { all_zero = 0; break; }
    }
    EXPECT(all_zero, "empty input → all-zero features");
}

/* ======================================================================== */
/* AST MUTATION TESTS                                                       */
/* ======================================================================== */

static ASTNode *parse_dsl_local(const char *src) {
    Token *tokens = (Token *)ALIEN_MALLOC(sizeof(Token) * 1024);
    if (!tokens) return NULL;
    int n_tok = tokenize(src, tokens, 1024);
    if (n_tok < 0) { ALIEN_FREE(tokens, sizeof(Token) * 1024); return NULL; }
    ASTNode *root = parse(tokens, n_tok);
    ALIEN_FREE(tokens, sizeof(Token) * 1024);
    return root;
}

static void test_ast_basics(void) {
    section("AST: parse / render / copy");

    /* Round-trip: parse → render → re-parse should give the same size. */
    const char *exprs[] = {
        "(seq 60 - 64 - 67 -)",
        "(euclid 5 8)",
        "(rotate (seq 60 64 67 72) 2)",
        "(mask (seq 60 64 67 72) (euclid 4 16))",
        "(drunk 16 3 60)",
    };
    for (int i = 0; i < 5; i++) {
        ASTNode *a = parse_dsl_local(exprs[i]);
        EXPECT(a != NULL, "parse known expression");
        if (!a) continue;

        char buf[1024];
        EXPECT(ns_ast_render(a, buf, 1024), "render OK");

        ASTNode *b = parse_dsl_local(buf);
        EXPECT(b != NULL, "re-parse rendered string");
        if (b) {
            EXPECT(ns_ast_size(a) == ns_ast_size(b), "round-trip size matches");
            ast_free(b);
        }
        ASTNode *c = ns_ast_copy(a);
        EXPECT(c != NULL, "copy succeeds");
        if (c) {
            EXPECT(ns_ast_size(a) == ns_ast_size(c), "copy preserves size");
            EXPECT(ns_ast_depth(a) == ns_ast_depth(c), "copy preserves depth");
            ast_free(c);
        }
        ast_free(a);
    }
}

static void test_ast_gen(void) {
    section("AST: gen_tree");

    ns_rng_t r;
    ns_rng_seed(&r, 1234);
    int total_size = 0, max_size = 0, valid = 0;
    for (int i = 0; i < 100; i++) {
        ASTNode *t = ns_gen_tree(&r, NS_MAX_AST_DEPTH, 0);
        if (!t) continue;
        valid++;
        int sz = ns_ast_size(t);
        total_size += sz;
        if (sz > max_size) max_size = sz;

        char buf[2048];
        EXPECT(ns_ast_render(t, buf, 2048), "render gen_tree output");
        ASTNode *re = parse_dsl_local(buf);
        EXPECT(re != NULL, "re-parse gen_tree output");
        if (re) ast_free(re);
        ast_free(t);
    }
    EXPECT(valid >= 95, "≥95% of gen_tree calls succeed");
    printf("  gen_tree: %d/%d valid, mean size=%.1f, max size=%d\n",
           valid, 100, valid > 0 ? (double)total_size / valid : 0.0, max_size);
}

static void test_ast_mutate(void) {
    section("AST: mutate");

    ns_rng_t r;
    ns_rng_seed(&r, 5678);

    ASTNode *parent = parse_dsl_local("(seq 60 64 67 72)");
    EXPECT(parent != NULL, "parse parent");
    if (!parent) return;

    int n_diff = 0;
    int n_valid = 0;
    char parent_buf[1024];
    ns_ast_render(parent, parent_buf, 1024);

    for (int i = 0; i < 200; i++) {
        ASTNode *child = ns_mutate(&r, parent, 0.4f, 0);
        if (!child) continue;
        n_valid++;

        char buf[2048];
        if (ns_ast_render(child, buf, 2048)) {
            if (strcmp(buf, parent_buf) != 0) n_diff++;
            ASTNode *re = parse_dsl_local(buf);
            EXPECT(re != NULL, "mutate output re-parses");
            if (re) ast_free(re);
        }
        ast_free(child);
    }
    EXPECT(n_valid >= 195, "≥97.5% of mutate calls succeed");
    EXPECT(n_diff > 100, "majority of mutations actually change the tree");
    printf("  mutate: %d/200 valid, %d/200 differed from parent\n", n_valid, n_diff);
    ast_free(parent);
}

/* Lineage test: simulate the feedback loop. Start with a tiny seed, mutate,
 * keep the offspring as the new parent for the next iteration, and verify
 * AST size grows on average. This is what ns_alien_evo2.pd should achieve. */
static void test_ast_lineage_growth(void) {
    section("AST: lineage growth (complexification)");

    ns_rng_t r;
    ns_rng_seed(&r, 3141);

    ASTNode *parent = parse_dsl_local("(seq 60 - 64 - 67 -)");
    int seed_size = ns_ast_size(parent);
    EXPECT(parent != NULL, "parse seed");
    if (!parent) return;

    int N = 200;
    int sizes[200];
    int valid = 0;
    for (int i = 0; i < N; i++) {
        ASTNode *child = ns_mutate(&r, parent, 0.3f, 0);
        if (!child) continue;
        int sz = ns_ast_size(child);
        int dp = ns_ast_depth(child);
        if (sz > NS_MAX_AST_SIZE || dp > NS_MAX_AST_DEPTH) {
            ast_free(child);
            continue;
        }
        sizes[valid++] = sz;
        /* Render → re-parse round-trip to ensure the offspring is valid DSL. */
        char buf[2048];
        if (!ns_ast_render(child, buf, 2048)) { ast_free(child); continue; }
        ASTNode *re = parse_dsl_local(buf);
        if (!re) { ast_free(child); continue; }
        ast_free(re);
        /* Promote child to parent for next iteration — the lineage step. */
        ast_free(parent);
        parent = child;
    }
    /* With real lineage growth, late-stage mutations push trees against the
     * size/depth caps and get rejected — so a lower survival rate is expected. */
    EXPECT(valid >= 100, "≥50% of mutations survive bounds & re-parse");

    /* Compare mean size in the early vs late windows. */
    int early_n = valid / 4, late_start = 3 * valid / 4;
    if (early_n > 0 && valid - late_start > 0) {
        double early = 0, late = 0;
        for (int i = 0; i < early_n; i++) early += sizes[i];
        for (int i = late_start; i < valid; i++) late += sizes[i];
        double e = early / early_n;
        double l = late / (valid - late_start);
        printf("  seed size=%d, valid=%d/200, early=%.1f, late=%.1f, growth=%.1fx\n",
               seed_size, valid, e, l, l / (e > 0 ? e : 1.0));
        EXPECT(l > e, "AST size grows over a 200-step lineage");
    }
    ast_free(parent);
}

static void test_ast_crossover(void) {
    section("AST: crossover");

    ns_rng_t r;
    ns_rng_seed(&r, 9999);

    ASTNode *a = parse_dsl_local("(seq 60 - 64 - 67 -)");
    ASTNode *b = parse_dsl_local("(rotate (seq 60 64 67 72) 2)");
    EXPECT(a && b, "parse both parents");
    if (!a || !b) { if (a) ast_free(a); if (b) ast_free(b); return; }

    int n_valid = 0;
    for (int i = 0; i < 100; i++) {
        ASTNode *c = ns_crossover(a, b, &r);
        if (!c) continue;
        n_valid++;
        char buf[2048];
        EXPECT(ns_ast_render(c, buf, 2048), "render crossover output");
        ASTNode *re = parse_dsl_local(buf);
        EXPECT(re != NULL, "crossover output re-parses");
        if (re) ast_free(re);
        ast_free(c);
    }
    EXPECT(n_valid >= 98, "crossover succeeds nearly always");
    ast_free(a); ast_free(b);
}

/* ======================================================================== */
/* INTEGRATION SMOKE TEST                                                   */
/* ======================================================================== */

static void test_integration(void) {
    section("integration: 4x4 random search → archive");

    ns_rng_t rng;
    ns_rng_seed(&rng, 2024);

    ns_archive_t *raw_archive = ns_archive_create();    /* 16-dim raw bitmap BC */
    ns_archive_t *stat_archive = ns_archive_create();   /* 6-dim hand-engineered BC */

    int iters = 500;
    float novelties_raw[500];
    float novelties_stat[500];

    for (int it = 0; it < iters; it++) {
        float grid[16];
        for (int i = 0; i < 16; i++) {
            grid[i] = (ns_rng_uniform(&rng) < 0.5f) ? 0.0f : 1.0f;
        }
        float stats[6];
        ns_grid_stats(grid, 4, 4, stats);

        novelties_raw[it]  = ns_archive_score(raw_archive, grid, 16);
        novelties_stat[it] = ns_archive_score(stat_archive, stats, 6);

        ns_archive_add(raw_archive, grid, 16);
        ns_archive_add(stat_archive, stats, 6);
    }

    /* The stat archive's novelty should drop over time (saturation).
     * The raw 16-dim bitmap BC, by contrast, should stay nearly flat —
     * this is the curse-of-dimensionality finding (H1 vs H2). */
    double early_s = 0.0, late_s = 0.0, early_r = 0.0, late_r = 0.0;
    int ec = 0, lc = 0;
    for (int i = 1; i < 100; i++) { /* skip iter 0: +inf */
        if (isfinite(novelties_stat[i])) { early_s += novelties_stat[i]; ec++; }
        if (isfinite(novelties_raw[i]))  { early_r += novelties_raw[i]; }
    }
    for (int i = 400; i < 500; i++) {
        if (isfinite(novelties_stat[i])) { late_s += novelties_stat[i]; lc++; }
        if (isfinite(novelties_raw[i]))  { late_r += novelties_raw[i]; }
    }
    EXPECT(ec > 0 && lc > 0, "novelty samples in both windows");
    if (ec > 0 && lc > 0) {
        double es = early_s / ec, ls = late_s / lc;
        double er = early_r / ec, lr = late_r / lc;
        printf("  stat BC (6d):  early=%.4f  late=%.4f  drop=%.1f%%\n",
               es, ls, 100.0 * (es - ls) / es);
        printf("  raw BC  (16d): early=%.4f  late=%.4f  drop=%.1f%%\n",
               er, lr, 100.0 * (er - lr) / er);
        EXPECT(ls < es, "stat-BC novelty decreases as archive fills");
    }

    EXPECT(raw_archive->count == iters, "raw archive count");
    EXPECT(stat_archive->count == iters, "stat archive count");

    ns_archive_destroy(raw_archive);
    ns_archive_destroy(stat_archive);
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

static void usage(void) {
    fprintf(stderr,
        "ns_parser %s — novelty-search core test/CLI\n\n"
        "Usage:\n"
        "  ns_parser --test                    run the full test suite\n"
        "  ns_parser --grid W H v0 v1 ...      print 6-dim stats for a flat grid\n"
        "  ns_parser --rng <seed> <count>      print <count> uniform samples\n",
        NS_VERSION_STRING);
}

static int cmd_test(void) {
    g_tests_run = 0;
    g_tests_failed = 0;
    test_rng();
    test_distance();
    test_archive_basic();
    test_archive_grow();
    test_archive_persistence();
    test_mutation();
    test_grid_stats();
    test_seq_features();
    test_ast_basics();
    test_ast_gen();
    test_ast_mutate();
    test_ast_lineage_growth();
    test_ast_crossover();
    test_integration();
    printf("\n");
    printf("==================================================\n");
    printf("  %d tests, %d failed\n", g_tests_run, g_tests_failed);
    printf("==================================================\n");
    return g_tests_failed > 0 ? 1 : 0;
}

static int cmd_grid(int argc, char **argv) {
    if (argc < 4) { usage(); return 2; }
    int w = atoi(argv[2]);
    int h = atoi(argv[3]);
    int n = w * h;
    if (argc < 4 + n) {
        fprintf(stderr, "expected %d cells after dims\n", n);
        return 2;
    }
    float *grid = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) grid[i] = (float)atof(argv[4 + i]);
    float out[6];
    ns_grid_stats(grid, w, h, out);
    printf("density=%.6f entropy=%.6f sym_h=%.6f sym_v=%.6f comps=%.6f mean_comp=%.6f\n",
           out[0], out[1], out[2], out[3], out[4], out[5]);
    free(grid);
    return 0;
}

static int cmd_rng(int argc, char **argv) {
    if (argc < 4) { usage(); return 2; }
    uint64_t seed = (uint64_t)strtoull(argv[2], NULL, 10);
    int count = atoi(argv[3]);
    ns_rng_t r;
    ns_rng_seed(&r, seed);
    for (int i = 0; i < count; i++) printf("%.9f\n", ns_rng_uniform(&r));
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    if (strcmp(argv[1], "--test") == 0) return cmd_test();
    if (strcmp(argv[1], "--grid") == 0) return cmd_grid(argc, argv);
    if (strcmp(argv[1], "--rng") == 0)  return cmd_rng(argc, argv);
    usage();
    return 2;
}
