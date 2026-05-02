/*
 * validate_seeds - Audit a seeds.txt file for ns_corpus consumption.
 *
 *   ./validate_seeds seeds.txt
 *   ./validate_seeds --verbose seeds.txt
 *   ./validate_seeds --strict seeds.txt   (exit 1 on any failure)
 *
 * For each non-blank, non-comment line, runs the same pipeline a runtime
 * loader would: tokenize → parse → eval → degenerate-check on the rendered
 * output. Reports counts and (in --verbose mode) per-failure detail.
 *
 * Degenerate criteria mirror what ns_quality will reject as score=0:
 *   - rendered length < 4
 *   - all rests
 *
 * Note: low variety (e.g. `(rep 42 16)` or `(euclid 5 8)`) is intentionally
 * NOT a hard reject. Single-value rhythmic patterns are legitimate seeds.
 * Variety contributes to the *soft* score in ns_quality, not the gate.
 *
 * Exit codes:
 *   0 — all entries valid (or --strict not set and parse succeeded for the file)
 *   1 — one or more failures and --strict set
 *   2 — could not open file / IO error
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../alien_core.h"

#define MAX_LINE 4096
#define MIN_OUTPUT_LEN 4

typedef enum {
    R_OK = 0,
    R_PARSE_FAIL,
    R_RENDER_FAIL,
    R_DEGENERATE,
} result_t;

typedef struct {
    int line_no;
    result_t code;
    char text[MAX_LINE];
    char detail[256];
} failure_t;

static const char *result_name(result_t r) {
    switch (r) {
        case R_OK:           return "ok";
        case R_PARSE_FAIL:   return "parse_fail";
        case R_RENDER_FAIL:  return "render_fail";
        case R_DEGENERATE:   return "degenerate";
    }
    return "?";
}

/* Strip leading/trailing whitespace and a trailing semicolon (the
 * Pd-message convention used in patterns.txt). Modifies in place. */
static void normalize(char *s) {
    /* trim trailing */
    int n = (int)strlen(s);
    while (n > 0 && (isspace((unsigned char)s[n-1]) || s[n-1] == ';')) {
        s[--n] = '\0';
    }
    /* trim leading */
    int i = 0;
    while (s[i] && isspace((unsigned char)s[i])) i++;
    if (i > 0) memmove(s, s + i, strlen(s + i) + 1);
}

/* True if line is blank or starts with #. */
static int is_skippable(const char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return *s == '\0' || *s == '#';
}

/* Inspect a rendered Sequence and return R_OK / R_DEGENERATE.
 * A "degenerate" output:
 *   - has fewer than MIN_OUTPUT_LEN elements
 *   - is entirely rests */
static result_t classify_output(const Sequence *seq, char *detail, int detail_len) {
    if (!seq) {
        snprintf(detail, detail_len, "null sequence");
        return R_DEGENERATE;
    }
    if (seq->length < MIN_OUTPUT_LEN) {
        snprintf(detail, detail_len, "length %d < %d", seq->length, MIN_OUTPUT_LEN);
        return R_DEGENERATE;
    }
    int rest_count = 0;
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == ALIEN_REST) rest_count++;
    }
    if (rest_count == seq->length) {
        snprintf(detail, detail_len, "all rests (%d)", seq->length);
        return R_DEGENERATE;
    }
    return R_OK;
}

/* Run one expression through the full pipeline. Returns the result and
 * fills detail with the alien_core error message on parse/render failures,
 * or with degenerate-criteria info on degenerate outputs. */
static result_t check_one(const char *expr, char *detail, int detail_len) {
    Token tokens[2048];
    int ntok = tokenize(expr, tokens, 2048);
    if (ntok < 0) {
        snprintf(detail, detail_len, "%s", g_error_message);
        return R_PARSE_FAIL;
    }
    ASTNode *ast = parse(tokens, ntok);
    if (!ast) {
        snprintf(detail, detail_len, "%s", g_error_message);
        return R_PARSE_FAIL;
    }
    /* Reset eval depth in case a previous deep eval bailed out mid-tree. */
    g_eval_depth = 0;
    Sequence *seq = eval_node(ast);
    ast_free(ast);
    if (!seq) {
        snprintf(detail, detail_len, "%s", g_error_message);
        return R_RENDER_FAIL;
    }
    result_t r = classify_output(seq, detail, detail_len);
    seq_free(seq);
    return r;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--verbose] [--strict] <seeds.txt>\n"
        "  --verbose   print each failure with its source line and reason\n"
        "  --strict    exit 1 if any failures (default: exit 0 unless IO error)\n",
        prog);
}

int main(int argc, char **argv) {
    int verbose = 0;
    int strict = 0;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--strict") == 0) {
            strict = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown flag: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        } else {
            path = argv[i];
        }
    }
    if (!path) {
        usage(argv[0]);
        return 2;
    }

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "validate_seeds: cannot open %s\n", path);
        return 2;
    }

    /* Make alien_core's stochastic ops deterministic across a run. The
     * specific seed doesn't matter — just don't reseed mid-pipeline. */
    g_random_initialized = true;
    srand(1);

    int total = 0, skipped = 0;
    int counts[4] = {0, 0, 0, 0};

    /* Up to 64 failures are remembered for verbose output. Beyond that
     * we still count but don't store. */
    failure_t failures[64];
    int nfail = 0;

    char line[MAX_LINE];
    int line_no = 0;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        if (is_skippable(line)) {
            skipped++;
            continue;
        }
        normalize(line);
        if (line[0] == '\0') {
            skipped++;
            continue;
        }
        total++;

        char detail[256] = {0};
        result_t r = check_one(line, detail, sizeof(detail));
        counts[r]++;

        if (r != R_OK && nfail < 64) {
            failures[nfail].line_no = line_no;
            failures[nfail].code = r;
            snprintf(failures[nfail].text, MAX_LINE, "%s", line);
            snprintf(failures[nfail].detail, sizeof(failures[nfail].detail),
                     "%s", detail);
            nfail++;
        }
    }
    fclose(f);

    if (verbose && nfail > 0) {
        for (int i = 0; i < nfail; i++) {
            fprintf(stderr, "  line %d (%s): %s\n      → %s\n",
                    failures[i].line_no,
                    result_name(failures[i].code),
                    failures[i].text,
                    failures[i].detail);
        }
    }

    int ok = counts[R_OK];
    int fails = counts[R_PARSE_FAIL] + counts[R_RENDER_FAIL] + counts[R_DEGENERATE];

    printf("%s: %d entries (%d skipped)\n", path, total, skipped);
    printf("  ok           %d\n", ok);
    printf("  parse_fail   %d\n", counts[R_PARSE_FAIL]);
    printf("  render_fail  %d\n", counts[R_RENDER_FAIL]);
    printf("  degenerate   %d\n", counts[R_DEGENERATE]);
    if (total > 0) {
        printf("  pass rate    %.1f%%\n", 100.0 * ok / total);
    }

    if (strict && fails > 0) return 1;
    return 0;
}
