/*
 * alien_parser - Standalone CLI pattern generator
 *
 * Tests and demonstrates the alien pattern language outside of Pure Data.
 * Can be used as a command-line tool or to run the test suite.
 *
 * Usage:
 *   ./alien_parser "(euclid 5 8)"          - Process single expression
 *   echo "(seq 1 2 3)" | ./alien_parser    - Read from stdin
 *   ./alien_parser --test                  - Run test suite
 */

#include <stdio.h>
#include "alien_core.h"

// ============================================================================
// CLI INTERFACE
// ============================================================================

int process_input(const char *input, char *output, size_t output_size) {
    if (!input || strlen(input) == 0) {
        set_error("Empty input");
        return -1;
    }

    Token tokens[1024];
    int token_count = tokenize(input, tokens, 1024);
    if (token_count < 0) return -1;

    ASTNode *ast = parse(tokens, token_count);
    if (!ast) return -1;

    Sequence *result = eval_node(ast);
    if (!result) {
        ast_free(ast);
        return -1;
    }

    // Format output
    char *p = output;
    size_t remaining = output_size;
    for (int i = 0; i < result->length && remaining > 1; i++) {
        if (i > 0) {
            *p++ = ' ';
            remaining--;
        }

        int written;
        if (result->values[i] == -1) {
            written = snprintf(p, remaining, "-");
        } else {
            written = snprintf(p, remaining, "%d", result->values[i]);
        }

        if (written > 0) {
            p += written;
            remaining -= written;
        }
    }
    *p = '\0';

    seq_free(result);
    ast_free(ast);
    return 0;
}

// ============================================================================
// TEST SYSTEM
// ============================================================================

typedef struct {
    const char *input;
    const char *expected_output;
    const char *description;
} TestCase;

TestCase tests[] = {
    // Basic values
    {"1", "1", "single number"},
    {"0", "0", "zero"},
    {"-", "-", "single hyphen"},

    // Seq operator - basic
    {"(seq 1 2 3)", "1 2 3", "simple sequence"},
    {"(seq 1 - 2)", "1 - 2", "sequence with rest"},

    // Rep operator
    {"(rep 1 3)", "1 1 1", "repeat single value"},
    {"(rep (seq 1 2) 3)", "1 2 1 2 1 2", "repeat sequence"},

    // Arithmetic
    {"(add (seq 1 2 3) 5)", "6 7 8", "add to sequence"},
    {"(mul (seq 1 2 3) 2)", "2 4 6", "multiply sequence"},
    {"(mod (seq 8 9 10) 7)", "1 2 3", "modulo sequence"},

    // Rhythm
    {"(euclid 3 8)", "- - 1 - - 1 - 1", "euclidean 3/8"},
    {"(euclid 3 8 2)", "1 - - 1 - 1 - -", "euclidean 3/8 rotated"},
    {"(subdiv (seq 1 2) 2)", "1 1 2 2", "subdivide by 2"},

    // List manipulation
    {"(reverse (seq 1 2 3))", "3 2 1", "reverse sequence"},
    {"(rotate (seq 1 2 3 4) 1)", "4 1 2 3", "rotate right"},
    {"(palindrome (seq 1 2 3))", "1 2 3 2 1", "palindrome"},
    {"(interleave (seq 1 2 3) (seq - - -))", "1 - 2 - 3 -", "interleave two sequences"},

    // Selection
    {"(take (seq 1 2 3 4 5) 3)", "1 2 3", "take first 3"},
    {"(drop (seq 1 2 3 4 5) 2)", "3 4 5", "drop first 2"},
    {"(every (seq 1 2 3 4 5 6) 2)", "1 3 5", "every 2nd element"},
    {"(filter (seq 1 - 2 - 3))", "1 2 3", "filter out hyphens"},

    // Pattern generation
    {"(range 1 5)", "1 2 3 4 5", "range 1 to 5"},
    {"(range 0 8 2)", "0 2 4 6 8", "range with step"},

    // Logic
    {"(cycle (seq 1 2 3) 8)", "1 2 3 1 2 3 1 2", "cycle pattern"},

    // Musical
    {"(transpose (seq 60 64 67) 5)", "65 69 72", "transpose up 5"},
    {"(chord 60 0)", "60 64 67", "C major chord"},

    // Time/phase
    {"(delay (seq 1 2 3) 2)", "- - 1 2 3", "delay by 2"},
    {"(gate (seq 1 2 3 4 5 6) 2)", "1 - 3 - 5 -", "gate every 2"},

    // Combined
    {"(seq (rep 1 2) (rep 2 2))", "1 1 2 2", "seq of reps"},
    {"(reverse (rep (seq 1 2) 3))", "2 1 2 1 2 1", "reverse repeated"},

    // Errors
    {"(seq 1 2", NULL, "unclosed parenthesis"},
    {"(unknown 1 2)", NULL, "unknown operator"},

    {NULL, NULL, NULL}
};

void run_tests(void) {
    int passed = 0;
    int failed = 0;

    printf("Running tests...\n\n");

    for (int i = 0; tests[i].input != NULL; i++) {
        printf("Test %d: %s\n", i + 1, tests[i].description);
        printf("  Input: '%s'\n", tests[i].input);

        char output[2048];
        int result = process_input(tests[i].input, output, sizeof(output));

        if (tests[i].expected_output == NULL) {
            // Expecting an error
            if (result < 0) {
                printf("  \033[32mPASS\033[0m (error as expected: %s)\n", g_error_message);
                passed++;
            } else {
                printf("  \033[31mFAIL\033[0m (expected error, got: '%s')\n", output);
                failed++;
            }
        } else {
            // Expecting success
            if (result >= 0 && strcmp(output, tests[i].expected_output) == 0) {
                printf("  \033[32mPASS\033[0m\n");
                passed++;
            } else if (result < 0) {
                printf("  \033[31mFAIL\033[0m (got error: %s)\n", g_error_message);
                printf("    Expected: '%s'\n", tests[i].expected_output);
                failed++;
            } else {
                printf("  \033[31mFAIL\033[0m\n");
                printf("    Expected: '%s'\n", tests[i].expected_output);
                printf("    Got:      '%s'\n", output);
                failed++;
            }
        }
        printf("\n");
    }

    printf("========================================\n");
    printf("Results: \033[32m%d passed\033[0m, ", passed);
    if (failed > 0) {
        printf("\033[31m%d failed\033[0m\n", failed);
    } else {
        printf("%d failed\n", failed);
    }
}

// ============================================================================
// MAIN
// ============================================================================

char* read_stdin(void) {
    static char buffer[8192];
    size_t len = 0;
    int c;

    while ((c = getchar()) != EOF && len < sizeof(buffer) - 1) {
        buffer[len++] = c;
    }
    buffer[len] = '\0';

    // Strip trailing newline
    if (len > 0 && buffer[len-1] == '\n') {
        buffer[len-1] = '\0';
    }

    return buffer;
}

int main(int argc, char **argv) {
    // Check for test mode
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        run_tests();
        return 0;
    }

    // Get input from command line or stdin
    const char *input;
    if (argc > 1) {
        input = argv[1];
    } else {
        input = read_stdin();
    }

    // Process and output
    char output[8192];
    int result = process_input(input, output, sizeof(output));

    if (result < 0) {
        fprintf(stderr, "Error: %s\n", g_error_message);
        return 1;
    }

    printf("%s\n", output);
    return 0;
}
