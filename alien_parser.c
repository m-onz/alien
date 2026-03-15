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
        if (result->values[i] == ALIEN_REST) {
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
    // ========================================================================
    // BASIC VALUES
    // ========================================================================
    {"1", "1", "single number"},
    {"0", "0", "zero"},
    {"-", "-", "single hyphen"},
    {"127", "127", "large number"},
    {"999", "999", "very large number"},

    // ========================================================================
    // SEQ OPERATOR
    // ========================================================================
    {"(seq 1 2 3)", "1 2 3", "seq: simple"},
    {"(seq 1 - 2)", "1 - 2", "seq: with rest"},
    {"(seq 60)", "60", "seq: single element"},
    {"(seq)", "", "seq: empty"},
    {"(seq - - -)", "- - -", "seq: all rests"},
    {"(seq 0 0 0)", "0 0 0", "seq: all zeros"},
    {"(seq (seq 1 2) (seq 3 4))", "1 2 3 4", "seq: nested seqs flatten"},

    // ========================================================================
    // REP OPERATOR
    // ========================================================================
    {"(rep 1 3)", "1 1 1", "rep: single value"},
    {"(rep (seq 1 2) 3)", "1 2 1 2 1 2", "rep: sequence"},
    {"(rep - 4)", "- - - -", "rep: rest repeated"},
    {"(rep (seq 1 2) 1)", "1 2", "rep: repeat once (identity)"},
    {"(rep 60 1)", "60", "rep: single value once"},
    {"(rep (seq) 3)", "", "rep: empty sequence"},

    // ========================================================================
    // ARITHMETIC OPERATORS
    // ========================================================================
    // add
    {"(add (seq 1 2 3) 5)", "6 7 8", "add: to sequence"},
    {"(add (seq 1 - 3) 10)", "11 - 13", "add: preserves rests"},
    {"(add (seq 0) 0)", "0", "add: zero to zero"},
    {"(add (seq 60 64 67) 12)", "72 76 79", "add: transpose up octave"},

    // sub
    {"(sub (seq 72 - 64) 12)", "60 - 52", "sub: from sequence"},
    {"(sub (seq 0) 1)", "-1", "sub: producing negative one (not rest)"},
    {"(sub (seq 72 - 64) 73)", "-1 - -9", "sub: mixed rests and negatives"},
    {"(sub (seq 10 20 30) 10)", "0 10 20", "sub: basic subtraction"},

    // mul
    {"(mul (seq 1 2 3) 2)", "2 4 6", "mul: basic"},
    {"(mul (seq 1 - 3) 5)", "5 - 15", "mul: preserves rests"},
    {"(mul (seq 1 2 3) 0)", "0 0 0", "mul: by zero"},
    {"(mul (seq 1 2 3) 1)", "1 2 3", "mul: by one (identity)"},

    // mod
    {"(mod (seq 8 9 10) 7)", "1 2 3", "mod: basic"},
    {"(mod (seq 7 14 21) 7)", "0 0 0", "mod: exact multiples"},
    {"(mod (seq 1 - 5) 3)", "1 - 2", "mod: preserves rests"},
    {"(mod (seq 0 1 2) 5)", "0 1 2", "mod: values smaller than modulus"},

    // ========================================================================
    // SCALE OPERATOR
    // ========================================================================
    {"(scale (seq 0 5 10) 0 10 0 100)", "0 50 100", "scale: 0-10 to 0-100"},
    {"(scale (seq 0 64 127) 0 127 0 100)", "0 50 100", "scale: MIDI to percent"},
    {"(scale (seq 1 - 3) 0 10 0 100)", "10 - 30", "scale: preserves rests"},
    {"(scale (seq 1 2 3) 0 0 0 10)", NULL, "scale: zero range (error)"},
    {"(scale (seq 0 5 10) 0 10 60 72)", "60 66 72", "scale: to MIDI range"},

    // ========================================================================
    // CLAMP OPERATOR
    // ========================================================================
    {"(clamp (seq 1 5 10) 3 8)", "3 5 8", "clamp: basic"},
    {"(clamp (seq 0 50 127) 20 100)", "20 50 100", "clamp: MIDI range"},
    {"(clamp (seq 1 - 10) 3 8)", "3 - 8", "clamp: preserves rests"},
    {"(clamp (seq 5 5 5) 5 5)", "5 5 5", "clamp: min equals max"},
    {"(clamp (seq 1 2 3) 0 127)", "1 2 3", "clamp: values already in range"},
    {"(clamp (seq 1 2) 3)", NULL, "clamp: too few args (error)"},

    // ========================================================================
    // EUCLID OPERATOR
    // ========================================================================
    {"(euclid 3 8)", "- - 1 - - 1 - 1", "euclid: 3/8"},
    {"(euclid 3 8 2)", "1 - - 1 - 1 - -", "euclid: 3/8 rotated"},
    {"(euclid 0 8)", "- - - - - - - -", "euclid: zero hits"},
    {"(euclid 4 4)", "1 1 1 1", "euclid: all hits"},
    {"(euclid 1 4)", "- - - 1", "euclid: single hit"},
    {"(euclid 5 8)", "- 1 - 1 1 - 1 1", "euclid: 5/8"},
    {"(euclid 2 8)", "- - - 1 - - - 1", "euclid: 2/8"},
    {"(euclid (seq 36 38 42) 8)", "- - 36 - - 38 - 42", "euclid: sequence as hits distributes values"},
    // euclid with hit_value (4th arg)
    {"(euclid 3 8 0 36)", "- - 36 - - 36 - 36", "euclid: hit_value replaces 1"},
    {"(euclid 4 4 0 60)", "60 60 60 60", "euclid: all hits with hit_value"},
    {"(euclid 2 4 1 99)", "99 - 99 -", "euclid: hit_value with rotation"},
    {"(euclid 1 1)", "1", "euclid: minimal 1/1"},
    {"(euclid 0 1)", "-", "euclid: zero hits one step"},
    {"(euclid 3 8 0 0)", "- - 0 - - 0 - 0", "euclid: hit_value zero"},

    // ========================================================================
    // SUBDIV OPERATOR
    // ========================================================================
    {"(subdiv (seq 1 2) 2)", "1 1 2 2", "subdiv: by 2"},
    {"(subdiv (seq 1 2 3) 3)", "1 1 1 2 2 2 3 3 3", "subdiv: by 3"},
    {"(subdiv (seq 1 2 3) 1)", "1 2 3", "subdiv: by 1 (identity)"},
    {"(subdiv (seq 60 - 64) 2)", "60 60 - - 64 64", "subdiv: preserves rests"},
    {"(subdiv (seq) 3)", "", "subdiv: empty sequence"},
    {"(subdiv (seq 1) 4)", "1 1 1 1", "subdiv: single element"},

    // ========================================================================
    // LIST MANIPULATION
    // ========================================================================
    // reverse
    {"(reverse (seq 1 2 3))", "3 2 1", "reverse: basic"},
    {"(reverse (seq 1))", "1", "reverse: single element"},
    {"(reverse (seq))", "", "reverse: empty"},
    {"(reverse (seq 1 - 3))", "3 - 1", "reverse: with rests"},

    // rotate
    {"(rotate (seq 1 2 3 4) 1)", "4 1 2 3", "rotate: right by 1"},
    {"(rotate (seq 1 2 3 4) 2)", "3 4 1 2", "rotate: right by 2"},
    {"(rotate (seq 1 2 3 4) 4)", "1 2 3 4", "rotate: full cycle (identity)"},
    {"(rotate (seq 1 2 3 4) 0)", "1 2 3 4", "rotate: by 0 (identity)"},
    {"(rotate (seq 1) 5)", "1", "rotate: single element any amount"},

    // interleave
    {"(interleave (seq 1 2 3) (seq - - -))", "1 - 2 - 3 -", "interleave: basic"},
    {"(interleave (seq 1 2) (seq 10 20 30))", "1 10 2 20 30", "interleave: different lengths appends remainder"},
    {"(interleave (seq 1) (seq 2))", "1 2", "interleave: single elements"},

    // shuffle — non-deterministic, just test error cases
    {"(shuffle (seq))", "", "shuffle: empty sequence"},

    // ========================================================================
    // SELECTION / FILTERING
    // ========================================================================
    // take
    {"(take (seq 1 2 3 4 5) 3)", "1 2 3", "take: first 3"},
    {"(take (seq 1 2 3) 0)", "", "take: zero elements"},
    {"(take (seq 1 2 3) 5)", "1 2 3", "take: more than length"},
    {"(take (seq 1 2 3) 3)", "1 2 3", "take: exact length"},
    {"(take (seq 1 - 3) 2)", "1 -", "take: includes rests"},

    // drop
    {"(drop (seq 1 2 3 4 5) 2)", "3 4 5", "drop: first 2"},
    {"(drop (seq 1 2 3) 10)", "", "drop: more than length"},
    {"(drop (seq 1 2 3) 0)", "1 2 3", "drop: zero (identity)"},
    {"(drop (seq 1 - 3 4) 1)", "- 3 4", "drop: rest in result"},

    // every
    {"(every (seq 1 2 3 4 5 6) 2)", "1 3 5", "every: 2nd element"},
    {"(every (seq 1) 1)", "1", "every: single element"},
    {"(every (seq 1 2 3 4 5 6) 3)", "1 4", "every: 3rd element"},
    {"(every (seq 1 2 3 4) 1)", "1 2 3 4", "every: 1 (identity)"},

    // slice
    {"(slice (seq 10 20 30 40 50) 1 3)", "20 30", "slice: middle"},
    {"(slice (seq 10 20 30 40 50) 0 2)", "10 20", "slice: from start"},
    {"(slice (seq 10 20 30 40 50) 3 5)", "40 50", "slice: to end"},
    {"(slice (seq 10 20 30) 0 0)", "", "slice: empty range"},
    {"(slice (seq 10 20 30) 1 10)", "20 30", "slice: end beyond length"},
    {"(slice (seq 1 - 3 - 5) 0 5)", "1 - 3 - 5", "slice: full range"},

    // filter
    {"(filter (seq 1 - 2 - 3))", "1 2 3", "filter: remove rests"},
    {"(filter (seq - - -))", "", "filter: all rests"},
    {"(filter (seq 1 2 3))", "1 2 3", "filter: no rests (identity)"},
    {"(filter (seq))", "", "filter: empty"},

    // ========================================================================
    // PATTERN GENERATION
    // ========================================================================
    // range
    {"(range 1 5)", "1 2 3 4 5", "range: ascending"},
    {"(range 0 8 2)", "0 2 4 6 8", "range: with step"},
    {"(range 5 1)", "", "range: descending without step"},
    {"(range 0 0)", "0", "range: single value"},
    {"(range 60 72)", "60 61 62 63 64 65 66 67 68 69 70 71 72", "range: MIDI octave"},
    {"(range 0 10 3)", "0 3 6 9", "range: step doesn't land on end"},
    {"(range 5 1 -1)", NULL, "range: negative step (hyphen parse error)"},

    // ramp
    {"(ramp 0 10 4)", "0 3 7 10", "ramp: basic interpolation"},
    {"(ramp 60 72 5)", "60 63 66 69 72", "ramp: MIDI range"},
    {"(ramp 0 0 3)", "0 0 0", "ramp: flat"},
    {"(ramp 10 0 3)", "10 5 0", "ramp: descending"},
    {"(ramp 0 100 2)", "0 100", "ramp: two points"},
    {"(ramp 0 12 7)", "0 2 4 6 8 10 12", "ramp: 7 points"},

    // ========================================================================
    // CONSTRAINTS
    // ========================================================================
    // wrap
    {"(wrap (seq 0 5 10 15 20) 0 12)", "0 5 10 3 8", "wrap: basic modulo wrapping"},
    {"(wrap (seq 13 25 37) 0 12)", "1 1 1", "wrap: repeated wrapping"},
    {"(wrap (seq 0 6 12) 0 12)", "0 6 0", "wrap: boundary values"},
    {"(wrap (seq 1 - 5) 0 10)", "1 - 5", "wrap: preserves rests"},
    {"(wrap (seq 1 2 3) 0 0)", NULL, "wrap: max must be > min (error)"},

    // fold
    {"(fold (seq 0 5 10 15 20) 0 10)", "0 5 10 5 0", "fold: reflect at boundaries"},
    {"(fold (seq 1 - 8) 0 10)", "1 - 8", "fold: preserves rests"},
    {"(fold (seq 5) 0 10)", "5", "fold: value in range"},
    {"(fold (seq 0 10) 0 10)", "0 10", "fold: boundary values"},
    {"(fold (seq 1 2 3) 5 5)", NULL, "fold: max must be > min (error)"},

    // ========================================================================
    // RANDOMNESS (non-deterministic — test errors + structure only)
    // ========================================================================
    {"(rand 4 60)", NULL, "rand: 2 args (error)"},
    {"(choose (seq))", "", "choose: empty seq returns empty"},

    // ========================================================================
    // CYCLE OPERATOR
    // ========================================================================
    {"(cycle (seq 1 2 3) 8)", "1 2 3 1 2 3 1 2", "cycle: extend pattern"},
    {"(cycle (seq) 5)", "", "cycle: empty sequence"},
    {"(cycle (seq 1 2 3) 3)", "1 2 3", "cycle: exact length"},
    {"(cycle (seq 1 2 3) 1)", "1", "cycle: shorter than pattern"},
    {"(cycle (seq 60 - 64) 6)", "60 - 64 60 - 64", "cycle: with rests"},

    // ========================================================================
    // GROW OPERATOR
    // ========================================================================
    {"(grow (seq 1 2 3))", "1 - - 1 2 - 1 2 3", "grow: 3 elements"},
    {"(grow (seq 1))", "1", "grow: single element"},
    {"(grow (seq 1 2))", "1 - 1 2", "grow: 2 elements"},
    {"(grow (seq 60 64 67 72))", "60 - - - 60 64 - - 60 64 67 - 60 64 67 72", "grow: MIDI chord build"},
    {"(grow (seq))", "", "grow: empty"},

    // ========================================================================
    // MUSICAL OPERATORS
    // ========================================================================
    // quantize
    {"(quantize (seq 61 63 66) (seq 0 2 4 5 7 9 11))", "60 62 65", "quantize: C major scale"},
    {"(quantize (seq 1 6 10) (seq 0 4 7 11))", "0 7 11", "quantize: pitch class small values"},
    {"(quantize (seq 60) (seq 0 4 7))", "60", "quantize: exact match"},
    {"(quantize (seq 62) (seq 0 4 7))", "60", "quantize: equidistant snaps down"},

    // arp
    {"(arp (seq 60) 2 5)", "60 60 60 60 60", "arp: single note up-down"},
    {"(arp (seq 60 64 67) 0 6)", "60 64 67 60 64 67", "arp: up"},
    {"(arp (seq 60 64 67) 1 6)", "67 64 60 67 64 60", "arp: down"},
    {"(arp (seq 60 64 67) 2 8)", "60 64 67 64 60 64 67 64", "arp: up-down"},
    {"(arp (seq) 0 5)", "", "arp: empty"},
    {"(arp (seq 60 64 67) 0 3)", "60 64 67", "arp: exact length"},
    {"(arp (seq 1 2) 2 6)", "1 2 1 2 1 2", "arp: up-down 2 elements"},

    // ========================================================================
    // TIME / PHASE OPERATORS
    // ========================================================================
    // gate
    {"(gate (seq 1 2 3 4 5 6) 2)", "1 - 3 - 5 -", "gate: every 2"},
    {"(gate (seq 1 2 3 4 5 6) 3)", "1 - - 4 - -", "gate: every 3"},
    {"(gate (seq 1 2 3) 1)", "1 2 3", "gate: every 1 (identity)"},
    {"(gate (seq 1 - 3 4) 2)", "1 - 3 -", "gate: with existing rests"},

    // speed
    {"(speed (seq 1 2 3) 4)", "1 - - - 2 - - - 3 - - -", "speed: by 4"},
    {"(speed (seq 1 2 3) 2)", "1 - 2 - 3 -", "speed: by 2"},
    {"(speed (seq 1 2 3) 1)", "1 2 3", "speed: by 1 (identity)"},
    {"(speed (seq 1 - 3) 2)", "1 - - - 3 -", "speed: with rests"},
    {"(speed (seq 60) 3)", "60 - -", "speed: single note"},

    // mask
    {"(mask (seq 60 64 67 72) (euclid 3 4))", "- 60 64 67", "mask: with euclidean gate"},
    {"(mask (seq 1 2 3) (seq 1 - 1 - 1))", "1 - 2 - 3", "mask: basic gating"},
    {"(mask (seq 1 2) (seq 1 1 1 1))", "1 2 1 2", "mask: cycling source"},
    {"(mask (seq 1 2 3) (seq))", "", "mask: empty gate"},
    {"(mask (seq 36 38) (euclid 2 4))", "- 36 - 38", "mask: MIDI with euclid"},

    // mirror
    {"(mirror (seq 1 2 3))", "1 2 3 2 1", "mirror: palindrome"},
    {"(mirror (seq 1))", "1", "mirror: single element"},
    {"(mirror (seq 1 2))", "1 2 1", "mirror: two elements"},
    {"(mirror (seq 60 64 67 72))", "60 64 67 72 67 64 60", "mirror: MIDI notes"},
    {"(mirror (seq))", "", "mirror: empty"},
    {"(mirror (seq 1 - 3))", "1 - 3 - 1", "mirror: with rests"},

    // delay
    {"(delay (seq 1 2 3) 2)", "- - 1 2 3", "delay: by 2"},
    {"(delay (seq 1 2 3) 0)", "1 2 3", "delay: by 0 (identity)"},
    {"(delay (seq 60) 4)", "- - - - 60", "delay: single note"},
    {"(delay (seq 1 - 3) 1)", "- 1 - 3", "delay: with rests"},
    {"(delay (seq) 3)", "- - -", "delay: empty sequence gets rests"},

    // ========================================================================
    // COMPOSITION / NESTING
    // ========================================================================
    {"(seq (rep 1 2) (rep 2 2))", "1 1 2 2", "compose: seq of reps"},
    {"(reverse (rep (seq 1 2) 3))", "2 1 2 1 2 1", "compose: reverse repeated"},
    {"(add (euclid 3 4) 35)", "- 36 36 36", "compose: euclid + add for MIDI"},
    {"(reverse (range 1 5))", "5 4 3 2 1", "compose: reverse range"},
    {"(take (cycle (seq 1 2 3) 10) 7)", "1 2 3 1 2 3 1", "compose: take from cycle"},
    {"(mirror (range 1 4))", "1 2 3 4 3 2 1", "compose: mirror range"},
    {"(mask (range 60 67) (euclid 3 8))", "- - 60 - - 61 - 62", "compose: mask range with euclid"},
    {"(speed (euclid 2 4) 2)", "- - 1 - - - 1 -", "compose: speed stretches euclid"},
    {"(filter (euclid 3 8))", "1 1 1", "compose: filter euclid removes rests"},
    {"(grow (euclid 2 4))", "- - - - - 1 - - - 1 - - - 1 - 1", "compose: grow euclid"},
    {"(euclid 3 8 0 36)", "- - 36 - - 36 - 36", "compose: euclid with custom hit value"},
    {"(mask (seq 36 38 42) (euclid 3 8))", "- - 36 - - 38 - 42", "compose: mask distributes values over euclid"},
    {"(mul (euclid 3 8) 36)", "- - 36 - - 36 - 36", "compose: mul euclid for hit value"},
    {"(add (mul (euclid 4 8) 36) 0)", "- 36 - 36 - 36 - 36", "compose: deep nesting"},

    // ========================================================================
    // ERROR CASES
    // ========================================================================
    {"(seq 1 2", NULL, "error: unclosed parenthesis"},
    {"(unknown 1 2)", NULL, "error: unknown operator"},
    {"1 2 3", NULL, "error: trailing tokens"},
    {"(seq 1 2) (seq 3 4)", NULL, "error: trailing expression"},
    {"", NULL, "error: empty input"},
    {"()", NULL, "error: empty parens"},
    {"(add 1)", NULL, "error: add needs 2 args"},
    {"(sub 1)", NULL, "error: sub needs 2 args"},
    {"(mul 1)", NULL, "error: mul needs 2 args"},
    {"(mod 1)", NULL, "error: mod needs 2 args"},
    {"(euclid 3)", NULL, "error: euclid needs 2+ args"},
    {"(subdiv (seq 1 2))", NULL, "error: subdiv needs 2 args"},
    {"(take (seq 1))", NULL, "error: take needs 2 args"},
    {"(drop (seq 1))", NULL, "error: drop needs 2 args"},
    {"(every (seq 1))", NULL, "error: every needs 2 args"},
    {"(slice (seq 1) 0)", NULL, "error: slice needs 3 args"},
    {"(scale (seq 1) 0 10)", NULL, "error: scale needs 5 args"},
    {"(clamp (seq 1) 0)", NULL, "error: clamp needs 3 args"},
    {"(wrap (seq 1) 0)", NULL, "error: wrap needs 3 args"},
    {"(fold (seq 1) 0)", NULL, "error: fold needs 3 args"},
    {"(arp (seq 1) 0)", NULL, "error: arp needs 3 args"},
    {"(gate (seq 1))", NULL, "error: gate needs 2 args"},
    {"(speed (seq 1))", NULL, "error: speed needs 2 args"},
    {"(mask (seq 1))", NULL, "error: mask needs 2 args"},
    {"(delay (seq 1))", NULL, "error: delay needs 2 args"},
    {"(cycle (seq 1))", NULL, "error: cycle needs 2 args"},
    {"(ramp 0 10)", NULL, "error: ramp needs 3 args"},
    {"(range)", NULL, "error: range needs 2+ args"},

    // ========================================================================
    // REST PROPAGATION (verify - passes through all operators)
    // ========================================================================
    {"(add (seq - - -) 5)", "- - -", "rest propagation: add"},
    {"(mul (seq - - -) 2)", "- - -", "rest propagation: mul"},
    {"(mod (seq - - -) 3)", "- - -", "rest propagation: mod"},
    {"(clamp (seq - 5 -) 0 10)", "- 5 -", "rest propagation: clamp"},
    {"(scale (seq - 5 -) 0 10 0 100)", "- 50 -", "rest propagation: scale"},
    {"(wrap (seq - 5 -) 0 10)", "- 5 -", "rest propagation: wrap"},
    {"(fold (seq - 5 -) 0 10)", "- 5 -", "rest propagation: fold"},
    {"(reverse (seq - 1 -))", "- 1 -", "rest propagation: reverse"},
    {"(subdiv (seq 1 -) 2)", "1 1 - -", "rest propagation: subdiv"},

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
    // Check for version flag
    if (argc > 1 && (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0)) {
        printf("alien_parser %s\n", ALIEN_VERSION_STRING);
        return 0;
    }

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
