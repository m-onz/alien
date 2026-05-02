/*
 * ns_system_test - controlled end-to-end novelty-search lineage
 *
 * A CLI program that runs the full pipeline in one process — no Pd, no
 * subprocess. Every component boundary is observable. Deterministic given
 * a seed.
 *
 *   PRODUCER   ns_mutate(parent, rate, &rng) → child AST
 *   RENDERER   eval_node(child)              → Sequence of ints + rests
 *   PROJECTOR  ns_seq_features(values, len)  → 27-dim BC
 *   MEMORY     ns_archive_score(bc), add()   → score, archive grows
 *   FILTER     decision = score >= threshold
 *   LOGGER     CSV trace + per-iteration stats
 *   FEEDBACK   if pass: parent ← child       (the lineage step)
 *
 * Output is CSV-friendly:
 *     iter,size,depth,seq_len,score,decision,expr
 *
 * Usage:
 *     ns_system_test [--seed N] [--iters N] [--threshold F] [--rate F]
 *                    [--seed-expr "(...)"] [--quiet]
 *
 * Defaults: seed=42, iters=200, threshold=0.5, rate=0.3,
 *           seed-expr="(seq 60 - 64 - 67 -)"
 *
 * Why this exists: unit tests prove the math; this proves the *composition*.
 * It's a regression test for the integrated pipeline. Any change to
 * mutate/render/project/score should reproduce the same trace given the
 * same seed — or document why not.
 */

#include "ns_core.h"
#include "ns_alien_ast.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================== */
/* DEFAULT PARAMETERS                                                       */
/* ======================================================================== */

static const uint64_t DEFAULT_SEED = 42;
static const int      DEFAULT_ITERS = 200;
static const float    DEFAULT_THRESHOLD = 0.5f;
static const float    DEFAULT_RATE = 0.3f;
static const char    *DEFAULT_SEED_EXPR = "(seq 60 - 64 - 67 -)";

/* ======================================================================== */
/* HELPERS                                                                  */
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

/* Convert alien's Sequence to int array suitable for ns_seq_features.
 * alien uses ALIEN_REST = INT_MIN; ns_seq_features expects NS_REST = INT_MIN.
 * Same sentinel, no conversion needed. */

/* ======================================================================== */
/* MAIN LOOP                                                                */
/* ======================================================================== */

typedef struct {
    uint64_t seed;
    int iters;
    float threshold;
    float rate;
    const char *seed_expr;
    int quiet;
} ts_config_t;

static void parse_args(int argc, char **argv, ts_config_t *cfg) {
    cfg->seed = DEFAULT_SEED;
    cfg->iters = DEFAULT_ITERS;
    cfg->threshold = DEFAULT_THRESHOLD;
    cfg->rate = DEFAULT_RATE;
    cfg->seed_expr = DEFAULT_SEED_EXPR;
    cfg->quiet = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg->iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            cfg->threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
            cfg->rate = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed-expr") == 0 && i + 1 < argc) {
            cfg->seed_expr = argv[++i];
        } else if (strcmp(argv[i], "--quiet") == 0) {
            cfg->quiet = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                "ns_system_test — end-to-end novelty-search lineage\n\n"
                "Usage:\n"
                "  ns_system_test [--seed N] [--iters N] [--threshold F] [--rate F]\n"
                "                 [--seed-expr \"(...)\"] [--quiet]\n\n"
                "Output CSV columns:\n"
                "  iter,size,depth,seq_len,score,decision,expr\n");
            exit(0);
        }
    }
}

int main(int argc, char **argv) {
    ts_config_t cfg;
    parse_args(argc, argv, &cfg);

    /* RNG seeding: two RNGs to deal with.
     *  1. ns_rng (xoshiro256**): drives mutation. Trivially seeded.
     *  2. C rand(): drives alien's stochastic operators (shuffle, prob,
     *     drunk, choose). alien_core.h's init_random() seeds with time(NULL)
     *     and won't reseed. We pre-empt by setting the init flag here and
     *     calling srand() with our seed — alien's later init_random() will
     *     see the flag set and skip. The shared static lives in our own
     *     translation unit (alien_core.h's globals are static), so we can
     *     touch g_random_initialized directly. */
    ns_rng_t rng;
    ns_rng_seed(&rng, cfg.seed);
    g_random_initialized = true;
    srand((unsigned int)cfg.seed);

    /* Archive: empty at start, grows as offspring are admitted. */
    ns_archive_t *archive = ns_archive_create();
    if (!archive) { fprintf(stderr, "OOM creating archive\n"); return 1; }

    /* Initial parent. */
    ASTNode *parent = parse_dsl_local(cfg.seed_expr);
    if (!parent) {
        fprintf(stderr, "could not parse seed expression: %s\n", cfg.seed_expr);
        ns_archive_destroy(archive);
        return 1;
    }

    /* Header. */
    if (!cfg.quiet) {
        fprintf(stderr,
            "# ns_system_test seed=%llu iters=%d threshold=%.3f rate=%.3f\n"
            "# seed_expr: %s\n",
            (unsigned long long)cfg.seed, cfg.iters, cfg.threshold, cfg.rate,
            cfg.seed_expr);
    }
    printf("iter,size,depth,seq_len,score,decision,expr\n");

    /* Per-iteration counters for end-of-run summary. */
    int n_pass = 0;
    int n_render_fail = 0;
    int n_mutate_fail = 0;
    int n_size_fail = 0;
    double sum_size = 0, sum_depth = 0, sum_score = 0;
    int score_samples = 0;
    int max_size_seen = 0, max_depth_seen = 0;

    /* The main loop — one iteration = one mutate+render+project+score+decide. */
    for (int it = 0; it < cfg.iters; it++) {

        /* PRODUCER: mutate the cached parent into a fresh child. */
        ASTNode *child = ns_mutate(&rng, parent, cfg.rate, 0);
        if (!child) { n_mutate_fail++; continue; }

        int sz = ns_ast_size(child);
        int dp = ns_ast_depth(child);
        if (sz > NS_MAX_AST_SIZE || dp > NS_MAX_AST_DEPTH) {
            n_size_fail++;
            ast_free(child);
            continue;
        }
        if (sz > max_size_seen) max_size_seen = sz;
        if (dp > max_depth_seen) max_depth_seen = dp;
        sum_size += sz;
        sum_depth += dp;

        /* RENDERER: in-process eval via alien_core's eval_node. */
        Sequence *seq = eval_node(child);
        if (!seq || seq->length <= 0) {
            n_render_fail++;
            if (seq) seq_free(seq);
            ast_free(child);
            continue;
        }
        int seq_len = seq->length;

        /* PROJECTOR: convert sequence to BC vector.
         * alien's ALIEN_REST and ns_seq_features's NS_REST are both INT_MIN. */
        float bc[NS_SEQ_FEATURE_DIM];
        ns_seq_features(seq->values, seq->length, bc);
        seq_free(seq);

        /* MEMORY: score against archive (mean kNN distance), then admit. */
        float score = ns_archive_score(archive, bc, NS_SEQ_FEATURE_DIM);
        if (!isfinite(score)) score = 1e30f;
        ns_archive_add(archive, bc, NS_SEQ_FEATURE_DIM);
        if (isfinite(score) && score < 1e29f) {
            sum_score += score;
            score_samples++;
        }

        /* FILTER: decision = pass iff score >= threshold. */
        int decision = (score >= cfg.threshold);

        /* LOGGER: CSV row. Render expression at the very end so it's the
         * exact tree that produced this score. */
        char expr_buf[2048];
        ns_ast_render(child, expr_buf, sizeof(expr_buf));
        printf("%d,%d,%d,%d,%.6f,%d,%s\n",
               it, sz, dp, seq_len, (double)score, decision, expr_buf);

        /* FEEDBACK: if novel, child becomes the next parent. Otherwise the
         * current parent stays put for the next mutation. */
        if (decision) {
            n_pass++;
            ast_free(parent);
            parent = child;
        } else {
            ast_free(child);
        }
    }

    /* End-of-run summary on stderr so stdout stays pure CSV. */
    if (!cfg.quiet) {
        fprintf(stderr,
            "# --- summary ---\n"
            "# iters_attempted: %d\n"
            "# mutate_failed:   %d  (NULL from ns_mutate — should be ~0)\n"
            "# size_rejected:   %d  (size > %d or depth > %d)\n"
            "# render_failed:   %d  (alien returned NULL or empty seq)\n"
            "# admitted:        %d  (passed novelty threshold)\n"
            "# admit_rate:      %.1f%%\n"
            "# mean_size:       %.2f\n"
            "# mean_depth:      %.2f\n"
            "# max_size:        %d\n"
            "# max_depth:       %d\n"
            "# mean_score:      %.4f  (excluding +inf seed iter)\n"
            "# archive_size:    %d\n",
            cfg.iters, n_mutate_fail,
            n_size_fail, NS_MAX_AST_SIZE, NS_MAX_AST_DEPTH,
            n_render_fail, n_pass,
            cfg.iters > 0 ? 100.0 * n_pass / cfg.iters : 0.0,
            cfg.iters > 0 ? sum_size / cfg.iters : 0.0,
            cfg.iters > 0 ? sum_depth / cfg.iters : 0.0,
            max_size_seen, max_depth_seen,
            score_samples > 0 ? sum_score / score_samples : 0.0,
            archive->count);
    }

    ast_free(parent);
    ns_archive_destroy(archive);
    return 0;
}
