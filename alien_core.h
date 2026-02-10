/*
 * alien_core.h - Shared pattern language implementation
 *
 * This header contains the core pattern parsing and evaluation engine
 * shared between the Pure Data external (alien.c) and the standalone
 * CLI tool (alien_parser.c).
 *
 * Memory allocation is abstracted to work with both Pure Data's
 * getbytes/freebytes and standard C malloc/free.
 */

#ifndef ALIEN_CORE_H
#define ALIEN_CORE_H

// Version information
#define ALIEN_VERSION_MAJOR 0
#define ALIEN_VERSION_MINOR 2
#define ALIEN_VERSION_PATCH 1
#define ALIEN_VERSION_STRING "0.2.1"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <math.h>

// ============================================================================
// MEMORY ALLOCATION ABSTRACTION
// ============================================================================

#ifdef PD
    // Pure Data memory functions
    #include "m_pd.h"
    #define ALIEN_MALLOC(size) getbytes(size)
    #define ALIEN_FREE(ptr, size) freebytes(ptr, size)
    #define ALIEN_REALLOC(ptr, old_size, new_size) resizebytes(ptr, old_size, new_size)
#else
    // Standard C memory functions
    #define ALIEN_MALLOC(size) malloc(size)
    #define ALIEN_FREE(ptr, size) free(ptr)
    #define ALIEN_REALLOC(ptr, old_size, new_size) realloc(ptr, new_size)
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef enum {
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_NUMBER,
    TOK_HYPHEN,
    TOK_SYMBOL,
    TOK_EOF
} TokenType;

// Maximum symbol length (operator names, etc.)
#define ALIEN_MAX_SYMBOL_LEN 128

typedef struct {
    TokenType type;
    union {
        int number;
        char symbol[ALIEN_MAX_SYMBOL_LEN];
    } value;
    int line;
    int column;
} Token;

typedef enum {
    NODE_NUMBER,
    NODE_HYPHEN,
    NODE_SEQ,
    NODE_REP,
    // Arithmetic
    NODE_ADD,
    NODE_MUL,
    NODE_MOD,
    NODE_SCALE,
    NODE_CLAMP,
    // Rhythm
    NODE_EUCLID,
    NODE_BJORK,
    NODE_SUBDIV,
    // List manipulation
    NODE_REVERSE,
    NODE_ROTATE,
    NODE_PALINDROME,
    NODE_MIRROR,
    NODE_INTERLEAVE,
    NODE_SHUFFLE,
    // Selection/filtering
    NODE_TAKE,
    NODE_DROP,
    NODE_EVERY,
    NODE_SLICE,
    NODE_FILTER,
    // Randomness
    NODE_CHOOSE,
    NODE_RAND,
    NODE_PROB,
    NODE_MAYBE,
    // Pattern generation
    NODE_RANGE,
    NODE_RAMP,
    NODE_DRUNK,
    // Conditional/logic
    NODE_CYCLE,
    NODE_GROW,
    NODE_DEGRADE,
    // Musical
    NODE_TRANSPOSE,
    NODE_QUANTIZE,
    NODE_CHORD,
    NODE_ARP,
    // Time/phase
    NODE_DELAY,
    NODE_GATE
} NodeType;

typedef struct ASTNode {
    NodeType type;
    union {
        int number;
        struct {
            struct ASTNode **children;
            int child_count;
            int child_capacity;
        } op;
    } data;
} ASTNode;

typedef struct {
    int *values;      // -1 represents hyphen
    int length;
    int capacity;
} Sequence;

// ============================================================================
// ERROR HANDLING
// ============================================================================

static char g_error_message[256] = {0};

static void set_error(const char *msg) {
    snprintf(g_error_message, sizeof(g_error_message), "%s", msg);
}

// ============================================================================
// RANDOM NUMBER UTILITIES
// ============================================================================

static bool g_random_initialized = false;

static void init_random(void) {
    if (!g_random_initialized) {
        srand(time(NULL));
        g_random_initialized = true;
    }
}

static int random_range(int min, int max) {
    init_random();
    if (max < min) { int t = min; min = max; max = t; }
    unsigned int range = (unsigned int)max - (unsigned int)min + 1;
    if (range == 0) return min;  // overflow case: full int range
    return min + (int)(rand() % range);
}

// ============================================================================
// SEQUENCE OPERATIONS
// ============================================================================

// Forward declaration
static void seq_free(Sequence *seq);

static Sequence* seq_new(void) {
    Sequence *seq = (Sequence*)ALIEN_MALLOC(sizeof(Sequence));
    if (!seq) return NULL;
    seq->capacity = 16;
    seq->length = 0;
    seq->values = (int*)ALIEN_MALLOC(sizeof(int) * seq->capacity);
    if (!seq->values) {
        ALIEN_FREE(seq, sizeof(Sequence));
        return NULL;
    }
    return seq;
}

static bool seq_append(Sequence *seq, int value) {
    if (seq->length >= seq->capacity) {
        int old_cap = seq->capacity;
        int new_cap = seq->capacity * 2;
        if (new_cap < old_cap) return false;  // overflow check
        int *new_values = (int*)ALIEN_REALLOC(seq->values,
            sizeof(int) * old_cap, sizeof(int) * new_cap);
        if (!new_values) return false;
        seq->values = new_values;
        seq->capacity = new_cap;
    }
    seq->values[seq->length++] = value;
    return true;
}

static bool seq_extend(Sequence *dest, Sequence *src) {
    for (int i = 0; i < src->length; i++) {
        if (!seq_append(dest, src->values[i])) return false;
    }
    return true;
}

static Sequence* seq_copy(Sequence *src) {
    if (!src) return NULL;
    Sequence *copy = seq_new();
    if (!copy) return NULL;
    if (!seq_extend(copy, src)) {
        seq_free(copy);
        return NULL;
    }
    return copy;
}

static void seq_free(Sequence *seq) {
    if (seq) {
        if (seq->values) ALIEN_FREE(seq->values, sizeof(int) * seq->capacity);
        ALIEN_FREE(seq, sizeof(Sequence));
    }
}

// ============================================================================
// MATHEMATICAL HELPERS
// ============================================================================

static void euclidean_rhythm(int hits, int steps, int *pattern) {
    if (hits >= steps) {
        for (int i = 0; i < steps; i++) pattern[i] = 1;
        return;
    }
    if (hits == 0) {
        for (int i = 0; i < steps; i++) pattern[i] = 0;
        return;
    }
    int bucket = 0;
    for (int i = 0; i < steps; i++) {
        bucket += hits;
        if (bucket >= steps) {
            bucket -= steps;
            pattern[i] = 1;
        } else {
            pattern[i] = 0;
        }
    }
}

// Proper Bjorklund algorithm implementation using Euclidean distribution
// Based on "The Euclidean Algorithm Generates Traditional Musical Rhythms" by Toussaint
static void bjorklund_rhythm(int hits, int steps, int *pattern) {
    if (steps <= 0) return;
    if (hits >= steps) {
        for (int i = 0; i < steps; i++) pattern[i] = 1;
        return;
    }
    if (hits <= 0) {
        for (int i = 0; i < steps; i++) pattern[i] = 0;
        return;
    }

    // Initialize groups: 'hits' groups of [1] and 'rests' groups of [0]
    int rests = steps - hits;

    // Use temporary arrays for the algorithm
    // Each "group" is stored as a sequence of bits
    // We'll use a 2D approach with fixed max size
    #define MAX_BJORK_STEPS 256
    int groups[MAX_BJORK_STEPS][MAX_BJORK_STEPS];
    int group_lens[MAX_BJORK_STEPS];
    int num_groups = steps;

    // Initialize: first 'hits' groups are [1], rest are [0]
    for (int i = 0; i < hits; i++) {
        groups[i][0] = 1;
        group_lens[i] = 1;
    }
    for (int i = hits; i < steps; i++) {
        groups[i][0] = 0;
        group_lens[i] = 1;
    }

    // Bjorklund's algorithm: repeatedly distribute remainder groups
    int num_ones = hits;
    int num_zeros = rests;

    while (num_zeros > 1) {
        int distribute = (num_ones < num_zeros) ? num_ones : num_zeros;

        // Append each of the last 'distribute' groups to the first 'distribute' groups
        for (int i = 0; i < distribute; i++) {
            int src_idx = num_ones + num_zeros - 1 - i;
            int dst_idx = i;
            // Append groups[src_idx] to groups[dst_idx]
            for (int j = 0; j < group_lens[src_idx]; j++) {
                groups[dst_idx][group_lens[dst_idx]++] = groups[src_idx][j];
            }
        }

        // Update counts
        if (num_ones < num_zeros) {
            num_zeros = num_zeros - num_ones;
            // num_ones stays the same
        } else {
            int temp = num_zeros;
            num_zeros = num_ones - num_zeros;
            num_ones = temp;
        }
        num_groups = num_ones + num_zeros;
    }

    // Flatten the groups into the output pattern
    int pos = 0;
    for (int i = 0; i < num_groups && pos < steps; i++) {
        for (int j = 0; j < group_lens[i] && pos < steps; j++) {
            pattern[pos++] = groups[i][j];
        }
    }
    #undef MAX_BJORK_STEPS
}

// ============================================================================
// TOKENIZER
// ============================================================================

static int tokenize(const char *input, Token *tokens, int max_tokens) {
    int token_count = 0;
    int line = 1;
    int column = 1;
    const char *p = input;

    while (*p && token_count < max_tokens) {
        while (*p && isspace(*p)) {
            if (*p == '\n') { line++; column = 1; }
            else { column++; }
            p++;
        }
        if (!*p) break;

        Token *tok = &tokens[token_count++];
        tok->line = line;
        tok->column = column;

        if (*p == '(') {
            tok->type = TOK_LPAREN;
            p++; column++;
        } else if (*p == ')') {
            tok->type = TOK_RPAREN;
            p++; column++;
        } else if (*p == '-' && !isdigit(*(p+1))) {
            tok->type = TOK_HYPHEN;
            p++; column++;
        } else if (isdigit(*p)) {
            tok->type = TOK_NUMBER;
            long num = 0;
            while (isdigit(*p)) {
                long digit = *p - '0';
                // Overflow check: ensure num * 10 + digit <= INT_MAX
                if (num > (INT_MAX - digit) / 10) {
                    set_error("Number too large (overflow)");
                    return -1;
                }
                num = num * 10 + digit;
                p++; column++;
            }
            tok->value.number = (int)num;
        } else if (isalpha(*p)) {
            tok->type = TOK_SYMBOL;
            int i = 0;
            while ((isalpha(*p) || isdigit(*p)) && i < ALIEN_MAX_SYMBOL_LEN - 1) {
                tok->value.symbol[i++] = *p++;
                column++;
            }
            tok->value.symbol[i] = '\0';
            // Check if symbol was truncated
            if (isalpha(*p) || isdigit(*p)) {
                set_error("Symbol too long (truncated)");
                return -1;
            }
        } else {
            set_error("Invalid character");
            return -1;
        }
    }

    if (token_count < max_tokens) {
        tokens[token_count].type = TOK_EOF;
        token_count++;
    }
    return token_count;
}

// ============================================================================
// AST OPERATIONS
// ============================================================================

static ASTNode* ast_new_number(int value) {
    ASTNode *node = (ASTNode*)ALIEN_MALLOC(sizeof(ASTNode));
    if (!node) return NULL;
    node->type = NODE_NUMBER;
    node->data.number = value;
    return node;
}

static ASTNode* ast_new_hyphen(void) {
    ASTNode *node = (ASTNode*)ALIEN_MALLOC(sizeof(ASTNode));
    if (!node) return NULL;
    node->type = NODE_HYPHEN;
    return node;
}

static ASTNode* ast_new_op(NodeType type) {
    ASTNode *node = (ASTNode*)ALIEN_MALLOC(sizeof(ASTNode));
    if (!node) return NULL;
    node->type = type;
    node->data.op.children = (ASTNode**)ALIEN_MALLOC(sizeof(ASTNode*) * 4);
    if (!node->data.op.children) {
        ALIEN_FREE(node, sizeof(ASTNode));
        return NULL;
    }
    node->data.op.child_count = 0;
    node->data.op.child_capacity = 4;
    return node;
}

static bool ast_add_child(ASTNode *parent, ASTNode *child) {
    if (parent->data.op.child_count >= parent->data.op.child_capacity) {
        int old_cap = parent->data.op.child_capacity;
        int new_cap = parent->data.op.child_capacity * 2;
        if (new_cap < old_cap) return false;  // overflow check
        ASTNode **new_children = (ASTNode**)ALIEN_REALLOC(parent->data.op.children,
            sizeof(ASTNode*) * old_cap, sizeof(ASTNode*) * new_cap);
        if (!new_children) return false;
        parent->data.op.children = new_children;
        parent->data.op.child_capacity = new_cap;
    }
    parent->data.op.children[parent->data.op.child_count++] = child;
    return true;
}

static void ast_free(ASTNode *node) {
    if (!node) return;
    if (node->type != NODE_NUMBER && node->type != NODE_HYPHEN) {
        for (int i = 0; i < node->data.op.child_count; i++) {
            ast_free(node->data.op.children[i]);
        }
        ALIEN_FREE(node->data.op.children, sizeof(ASTNode*) * node->data.op.child_capacity);
    }
    ALIEN_FREE(node, sizeof(ASTNode));
}

// ============================================================================
// PARSER
// ============================================================================

typedef struct {
    Token *tokens;
    int pos;
    int count;
} Parser;

static Token* parser_current(Parser *p) {
    if (p->pos < p->count) return &p->tokens[p->pos];
    return &p->tokens[p->count - 1];
}

static Token* parser_advance(Parser *p) {
    if (p->pos < p->count - 1) p->pos++;
    return parser_current(p);
}

static ASTNode* parse_expr(Parser *p);

static ASTNode* parse_list(Parser *p) {
    Token *tok = parser_current(p);
    if (tok->type != TOK_LPAREN) {
        set_error("Expected '('");
        return NULL;
    }
    parser_advance(p);
    tok = parser_current(p);

    if (tok->type != TOK_SYMBOL) {
        set_error("Expected operator name");
        return NULL;
    }

    ASTNode *node = NULL;
    const char *op = tok->value.symbol;

    if (strcmp(op, "seq") == 0) node = ast_new_op(NODE_SEQ);
    else if (strcmp(op, "rep") == 0) node = ast_new_op(NODE_REP);
    else if (strcmp(op, "add") == 0) node = ast_new_op(NODE_ADD);
    else if (strcmp(op, "mul") == 0) node = ast_new_op(NODE_MUL);
    else if (strcmp(op, "mod") == 0) node = ast_new_op(NODE_MOD);
    else if (strcmp(op, "scale") == 0) node = ast_new_op(NODE_SCALE);
    else if (strcmp(op, "clamp") == 0) node = ast_new_op(NODE_CLAMP);
    else if (strcmp(op, "euclid") == 0) node = ast_new_op(NODE_EUCLID);
    else if (strcmp(op, "bjork") == 0) node = ast_new_op(NODE_BJORK);
    else if (strcmp(op, "subdiv") == 0) node = ast_new_op(NODE_SUBDIV);
    else if (strcmp(op, "reverse") == 0) node = ast_new_op(NODE_REVERSE);
    else if (strcmp(op, "rotate") == 0) node = ast_new_op(NODE_ROTATE);
    else if (strcmp(op, "palindrome") == 0) node = ast_new_op(NODE_PALINDROME);
    else if (strcmp(op, "mirror") == 0) node = ast_new_op(NODE_MIRROR);
    else if (strcmp(op, "interleave") == 0) node = ast_new_op(NODE_INTERLEAVE);
    else if (strcmp(op, "shuffle") == 0) node = ast_new_op(NODE_SHUFFLE);
    else if (strcmp(op, "take") == 0) node = ast_new_op(NODE_TAKE);
    else if (strcmp(op, "drop") == 0) node = ast_new_op(NODE_DROP);
    else if (strcmp(op, "every") == 0) node = ast_new_op(NODE_EVERY);
    else if (strcmp(op, "slice") == 0) node = ast_new_op(NODE_SLICE);
    else if (strcmp(op, "filter") == 0) node = ast_new_op(NODE_FILTER);
    else if (strcmp(op, "choose") == 0) node = ast_new_op(NODE_CHOOSE);
    else if (strcmp(op, "rand") == 0) node = ast_new_op(NODE_RAND);
    else if (strcmp(op, "prob") == 0) node = ast_new_op(NODE_PROB);
    else if (strcmp(op, "maybe") == 0) node = ast_new_op(NODE_MAYBE);
    else if (strcmp(op, "range") == 0) node = ast_new_op(NODE_RANGE);
    else if (strcmp(op, "ramp") == 0) node = ast_new_op(NODE_RAMP);
    else if (strcmp(op, "drunk") == 0) node = ast_new_op(NODE_DRUNK);
    else if (strcmp(op, "cycle") == 0) node = ast_new_op(NODE_CYCLE);
    else if (strcmp(op, "grow") == 0) node = ast_new_op(NODE_GROW);
    else if (strcmp(op, "degrade") == 0) node = ast_new_op(NODE_DEGRADE);
    else if (strcmp(op, "transpose") == 0) node = ast_new_op(NODE_TRANSPOSE);
    else if (strcmp(op, "quantize") == 0) node = ast_new_op(NODE_QUANTIZE);
    else if (strcmp(op, "chord") == 0) node = ast_new_op(NODE_CHORD);
    else if (strcmp(op, "arp") == 0) node = ast_new_op(NODE_ARP);
    else if (strcmp(op, "delay") == 0) node = ast_new_op(NODE_DELAY);
    else if (strcmp(op, "gate") == 0) node = ast_new_op(NODE_GATE);
    else {
        set_error("Unknown operator");
        return NULL;
    }

    parser_advance(p);

    while (parser_current(p)->type != TOK_RPAREN) {
        if (parser_current(p)->type == TOK_EOF) {
            set_error("Unexpected end of input");
            ast_free(node);
            return NULL;
        }
        ASTNode *child = parse_expr(p);
        if (!child) {
            ast_free(node);
            return NULL;
        }
        if (!ast_add_child(node, child)) {
            set_error("Memory allocation failed");
            ast_free(child);
            ast_free(node);
            return NULL;
        }
    }

    parser_advance(p);
    return node;
}

static ASTNode* parse_expr(Parser *p) {
    Token *tok = parser_current(p);
    switch (tok->type) {
        case TOK_NUMBER: {
            ASTNode *node = ast_new_number(tok->value.number);
            parser_advance(p);
            return node;
        }
        case TOK_HYPHEN: {
            ASTNode *node = ast_new_hyphen();
            parser_advance(p);
            return node;
        }
        case TOK_LPAREN:
            return parse_list(p);
        default:
            set_error("Unexpected token");
            return NULL;
    }
}

static ASTNode* parse(Token *tokens, int token_count) {
    Parser parser = { tokens, 0, token_count };
    return parse_expr(&parser);
}

// ============================================================================
// EVALUATOR FORWARD DECLARATIONS
// ============================================================================

static Sequence* eval_node(ASTNode *node);

// ============================================================================
// EVALUATOR IMPLEMENTATIONS
// ============================================================================

static Sequence* eval_seq(ASTNode *node) {
    Sequence *result = seq_new();
    if (!result) return NULL;
    for (int i = 0; i < node->data.op.child_count; i++) {
        Sequence *child = eval_node(node->data.op.children[i]);
        if (!child) { seq_free(result); return NULL; }
        if (!seq_extend(result, child)) { seq_free(child); seq_free(result); return NULL; }
        seq_free(child);
    }
    return result;
}

static Sequence* eval_rep(ASTNode *node) {
    if (node->data.op.child_count < 2) {
        set_error("rep requires at least 2 arguments");
        return NULL;
    }
    int count_idx = node->data.op.child_count - 1;
    Sequence *count_seq = eval_node(node->data.op.children[count_idx]);
    if (!count_seq || count_seq->length != 1 || count_seq->values[0] < 0) {
        set_error("rep: last argument must be a single non-negative number");
        seq_free(count_seq);
        return NULL;
    }
    int repeat_count = count_seq->values[0];
    seq_free(count_seq);

    Sequence *to_repeat = seq_new();
    if (!to_repeat) return NULL;
    for (int i = 0; i < count_idx; i++) {
        Sequence *child = eval_node(node->data.op.children[i]);
        if (!child) { seq_free(to_repeat); return NULL; }
        if (!seq_extend(to_repeat, child)) { seq_free(child); seq_free(to_repeat); return NULL; }
        seq_free(child);
    }

    Sequence *result = seq_new();
    if (!result) { seq_free(to_repeat); return NULL; }
    for (int i = 0; i < repeat_count; i++) {
        if (!seq_extend(result, to_repeat)) { seq_free(to_repeat); seq_free(result); return NULL; }
    }
    seq_free(to_repeat);
    return result;
}

static Sequence* eval_add(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("add requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *delta_seq = eval_node(node->data.op.children[1]);
    if (!seq || !delta_seq) { seq_free(seq); seq_free(delta_seq); return NULL; }
    if (delta_seq->length != 1) { set_error("add: second arg must be single number"); seq_free(seq); seq_free(delta_seq); return NULL; }
    int delta = delta_seq->values[0];
    seq_free(delta_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i] == -1 ? -1 : seq->values[i] + delta)) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_mul(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("mul requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *factor_seq = eval_node(node->data.op.children[1]);
    if (!seq || !factor_seq) { seq_free(seq); seq_free(factor_seq); return NULL; }
    if (factor_seq->length != 1) { set_error("mul: second arg must be single number"); seq_free(seq); seq_free(factor_seq); return NULL; }
    int factor = factor_seq->values[0];
    seq_free(factor_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i] == -1 ? -1 : seq->values[i] * factor)) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_mod(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("mod requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *divisor_seq = eval_node(node->data.op.children[1]);
    if (!seq || !divisor_seq) { seq_free(seq); seq_free(divisor_seq); return NULL; }
    if (divisor_seq->length != 1 || divisor_seq->values[0] <= 0) { set_error("mod: second arg must be positive number"); seq_free(seq); seq_free(divisor_seq); return NULL; }
    int divisor = divisor_seq->values[0];
    seq_free(divisor_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i] == -1 ? -1 : seq->values[i] % divisor)) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_scale(ASTNode *node) {
    if (node->data.op.child_count != 5) { set_error("scale requires 5 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *from_min_seq = eval_node(node->data.op.children[1]);
    Sequence *from_max_seq = eval_node(node->data.op.children[2]);
    Sequence *to_min_seq = eval_node(node->data.op.children[3]);
    Sequence *to_max_seq = eval_node(node->data.op.children[4]);
    if (!seq || !from_min_seq || !from_max_seq || !to_min_seq || !to_max_seq) {
        seq_free(seq); seq_free(from_min_seq); seq_free(from_max_seq);
        seq_free(to_min_seq); seq_free(to_max_seq);
        return NULL;
    }
    if (from_min_seq->length != 1 || from_max_seq->length != 1 || to_min_seq->length != 1 || to_max_seq->length != 1) {
        set_error("scale: range args must be single numbers");
        seq_free(seq); seq_free(from_min_seq); seq_free(from_max_seq);
        seq_free(to_min_seq); seq_free(to_max_seq);
        return NULL;
    }
    int from_min = from_min_seq->values[0];
    int from_max = from_max_seq->values[0];
    int to_min = to_min_seq->values[0];
    int to_max = to_max_seq->values[0];
    seq_free(from_min_seq); seq_free(from_max_seq); seq_free(to_min_seq); seq_free(to_max_seq);
    if (from_max == from_min) {
        set_error("scale: from_min and from_max cannot be equal");
        seq_free(seq);
        return NULL;
    }
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == -1) {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            double normalized = (double)(seq->values[i] - from_min) / (from_max - from_min);
            int scaled = to_min + (int)(normalized * (to_max - to_min));
            if (!seq_append(result, scaled)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_clamp(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("clamp requires 3 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *min_seq = eval_node(node->data.op.children[1]);
    Sequence *max_seq = eval_node(node->data.op.children[2]);
    if (!seq || !min_seq || !max_seq) { seq_free(seq); seq_free(min_seq); seq_free(max_seq); return NULL; }
    if (min_seq->length != 1 || max_seq->length != 1) { set_error("clamp: min and max must be single numbers"); seq_free(seq); seq_free(min_seq); seq_free(max_seq); return NULL; }
    int min_val = min_seq->values[0];
    int max_val = max_seq->values[0];
    seq_free(min_seq); seq_free(max_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == -1) {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            int val = seq->values[i];
            if (val < min_val) val = min_val;
            if (val > max_val) val = max_val;
            if (!seq_append(result, val)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_euclid(ASTNode *node) {
    if (node->data.op.child_count < 2 || node->data.op.child_count > 3) { set_error("euclid requires 2 or 3 arguments"); return NULL; }

    Sequence *pattern_seq = eval_node(node->data.op.children[0]);
    Sequence *steps_seq = eval_node(node->data.op.children[1]);

    if (!pattern_seq || !steps_seq) { seq_free(pattern_seq); seq_free(steps_seq); return NULL; }
    if (steps_seq->length != 1) { set_error("euclid: steps must be single number"); seq_free(pattern_seq); seq_free(steps_seq); return NULL; }

    int steps = steps_seq->values[0];
    seq_free(steps_seq);

    int hits;
    int is_hit_count = (pattern_seq->length == 1 && pattern_seq->values[0] > 0);

    if (is_hit_count) {
        hits = pattern_seq->values[0];
        seq_free(pattern_seq);
        pattern_seq = seq_new();
        if (!pattern_seq) return NULL;
        if (!seq_append(pattern_seq, 1)) { seq_free(pattern_seq); return NULL; }
    } else {
        hits = pattern_seq->length;
    }

    int rotation = 0;
    if (node->data.op.child_count == 3) {
        Sequence *rot_seq = eval_node(node->data.op.children[2]);
        if (!rot_seq || rot_seq->length != 1) { set_error("euclid: rotation must be single number"); seq_free(rot_seq); seq_free(pattern_seq); return NULL; }
        rotation = rot_seq->values[0];
        seq_free(rot_seq);
    }

    int *euclid_pattern = (int*)ALIEN_MALLOC(sizeof(int) * steps);
    if (!euclid_pattern) { seq_free(pattern_seq); return NULL; }
    euclidean_rhythm(hits, steps, euclid_pattern);

    Sequence *result = seq_new();
    if (!result) { ALIEN_FREE(euclid_pattern, sizeof(int) * steps); seq_free(pattern_seq); return NULL; }
    int pattern_idx = 0;
    for (int i = 0; i < steps; i++) {
        int idx = (i + rotation) % steps;
        if (euclid_pattern[idx]) {
            if (!seq_append(result, pattern_seq->values[pattern_idx % pattern_seq->length])) { seq_free(result); ALIEN_FREE(euclid_pattern, sizeof(int) * steps); seq_free(pattern_seq); return NULL; }
            pattern_idx++;
        } else {
            if (!seq_append(result, -1)) { seq_free(result); ALIEN_FREE(euclid_pattern, sizeof(int) * steps); seq_free(pattern_seq); return NULL; }
        }
    }

    ALIEN_FREE(euclid_pattern, sizeof(int) * steps);
    seq_free(pattern_seq);
    return result;
}

static Sequence* eval_bjork(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("bjork requires 2 arguments"); return NULL; }
    Sequence *hits_seq = eval_node(node->data.op.children[0]);
    Sequence *steps_seq = eval_node(node->data.op.children[1]);
    if (!hits_seq || !steps_seq) { seq_free(hits_seq); seq_free(steps_seq); return NULL; }
    if (hits_seq->length != 1 || steps_seq->length != 1) { set_error("bjork: hits and steps must be single numbers"); seq_free(hits_seq); seq_free(steps_seq); return NULL; }
    int hits = hits_seq->values[0];
    int steps = steps_seq->values[0];
    seq_free(hits_seq); seq_free(steps_seq);
    if (steps > 256) { set_error("bjork: max 256 steps"); return NULL; }
    if (steps <= 0) { set_error("bjork: steps must be positive"); return NULL; }
    int *pattern = (int*)ALIEN_MALLOC(sizeof(int) * steps);
    if (!pattern) return NULL;
    bjorklund_rhythm(hits, steps, pattern);
    Sequence *result = seq_new();
    if (!result) { ALIEN_FREE(pattern, sizeof(int) * steps); return NULL; }
    for (int i = 0; i < steps; i++) {
        if (!seq_append(result, pattern[i] ? 1 : -1)) { seq_free(result); ALIEN_FREE(pattern, sizeof(int) * steps); return NULL; }
    }
    ALIEN_FREE(pattern, sizeof(int) * steps);
    return result;
}

static Sequence* eval_subdiv(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("subdiv requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1 || n_seq->values[0] <= 0) { set_error("subdiv: n must be positive number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        for (int j = 0; j < n; j++) {
            if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_reverse(ASTNode *node) {
    if (node->data.op.child_count != 1) { set_error("reverse requires 1 argument"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    if (!seq) return NULL;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = seq->length - 1; i >= 0; i--) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_rotate(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("rotate requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1) { set_error("rotate: n must be single number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    if (seq->length == 0) return seq;
    n = n % seq->length;
    if (n < 0) n += seq->length;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        int idx = (seq->length - n + i) % seq->length;
        if (!seq_append(result, seq->values[idx])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_palindrome(ASTNode *node) {
    if (node->data.op.child_count != 1) { set_error("palindrome requires 1 argument"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    if (!seq) return NULL;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    for (int i = seq->length - 2; i >= 0; i--) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_mirror(ASTNode *node) {
    if (node->data.op.child_count != 1) { set_error("mirror requires 1 argument"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    if (!seq) return NULL;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    for (int i = seq->length - 1; i >= 0; i--) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_interleave(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("interleave requires 2 arguments"); return NULL; }
    Sequence *seq1 = eval_node(node->data.op.children[0]);
    Sequence *seq2 = eval_node(node->data.op.children[1]);
    if (!seq1 || !seq2) { seq_free(seq1); seq_free(seq2); return NULL; }
    Sequence *result = seq_new();
    if (!result) { seq_free(seq1); seq_free(seq2); return NULL; }
    int max_len = seq1->length > seq2->length ? seq1->length : seq2->length;
    for (int i = 0; i < max_len; i++) {
        if (i < seq1->length) { if (!seq_append(result, seq1->values[i])) { seq_free(result); seq_free(seq1); seq_free(seq2); return NULL; } }
        if (i < seq2->length) { if (!seq_append(result, seq2->values[i])) { seq_free(result); seq_free(seq1); seq_free(seq2); return NULL; } }
    }
    seq_free(seq1); seq_free(seq2);
    return result;
}

static Sequence* eval_shuffle(ASTNode *node) {
    if (node->data.op.child_count != 1) { set_error("shuffle requires 1 argument"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    if (!seq) return NULL;
    Sequence *result = seq_copy(seq);
    if (!result) { seq_free(seq); return NULL; }
    for (int i = result->length - 1; i > 0; i--) {
        int j = random_range(0, i);
        int temp = result->values[i];
        result->values[i] = result->values[j];
        result->values[j] = temp;
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_take(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("take requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1 || n_seq->values[0] < 0) { set_error("take: n must be non-negative number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    int limit = n < seq->length ? n : seq->length;
    for (int i = 0; i < limit; i++) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_drop(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("drop requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1 || n_seq->values[0] < 0) { set_error("drop: n must be non-negative number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = n; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_every(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("every requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1 || n_seq->values[0] <= 0) { set_error("every: n must be positive number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i += n) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_slice(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("slice requires 3 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *start_seq = eval_node(node->data.op.children[1]);
    Sequence *end_seq = eval_node(node->data.op.children[2]);
    if (!seq || !start_seq || !end_seq) { seq_free(seq); seq_free(start_seq); seq_free(end_seq); return NULL; }
    if (start_seq->length != 1 || end_seq->length != 1) { set_error("slice: start and end must be single numbers"); seq_free(seq); seq_free(start_seq); seq_free(end_seq); return NULL; }
    int start = start_seq->values[0];
    int end = end_seq->values[0];
    seq_free(start_seq); seq_free(end_seq);
    if (start < 0) start = 0;
    if (end > seq->length) end = seq->length;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = start; i < end && i < seq->length; i++) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_filter(ASTNode *node) {
    if (node->data.op.child_count != 1) { set_error("filter requires 1 argument"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    if (!seq) return NULL;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] != -1) {
            if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_choose(ASTNode *node) {
    if (node->data.op.child_count == 0) { set_error("choose requires at least 1 argument"); return NULL; }
    int choice = random_range(0, node->data.op.child_count - 1);
    return eval_node(node->data.op.children[choice]);
}

static Sequence* eval_rand(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("rand requires 3 arguments"); return NULL; }
    Sequence *count_seq = eval_node(node->data.op.children[0]);
    Sequence *min_seq = eval_node(node->data.op.children[1]);
    Sequence *max_seq = eval_node(node->data.op.children[2]);
    if (!count_seq || !min_seq || !max_seq) { seq_free(count_seq); seq_free(min_seq); seq_free(max_seq); return NULL; }
    if (count_seq->length != 1 || min_seq->length != 1 || max_seq->length != 1) { set_error("rand: all arguments must be single numbers"); seq_free(count_seq); seq_free(min_seq); seq_free(max_seq); return NULL; }
    int count = count_seq->values[0];
    int min = min_seq->values[0];
    int max = max_seq->values[0];
    seq_free(count_seq); seq_free(min_seq); seq_free(max_seq);
    Sequence *result = seq_new();
    if (!result) return NULL;
    for (int i = 0; i < count; i++) {
        if (!seq_append(result, random_range(min, max))) { seq_free(result); return NULL; }
    }
    return result;
}

static Sequence* eval_prob(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("prob requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *prob_seq = eval_node(node->data.op.children[1]);
    if (!seq || !prob_seq) { seq_free(seq); seq_free(prob_seq); return NULL; }
    if (prob_seq->length != 1) { set_error("prob: probability must be single number (0-100)"); seq_free(seq); seq_free(prob_seq); return NULL; }
    int prob = prob_seq->values[0];
    seq_free(prob_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        // Use 0-99 range (100 values) so prob=100 always triggers, prob=0 never triggers
        if (random_range(0, 99) < prob) {
            if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_maybe(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("maybe requires 3 arguments"); return NULL; }
    Sequence *prob_seq = eval_node(node->data.op.children[2]);
    if (!prob_seq || prob_seq->length != 1) { set_error("maybe: probability must be single number (0-100)"); seq_free(prob_seq); return NULL; }
    int prob = prob_seq->values[0];
    seq_free(prob_seq);
    // Use 0-99 range (100 values) so prob=100 always triggers, prob=0 never triggers
    if (random_range(0, 99) < prob) {
        return eval_node(node->data.op.children[0]);
    } else {
        return eval_node(node->data.op.children[1]);
    }
}

static Sequence* eval_range(ASTNode *node) {
    if (node->data.op.child_count < 2 || node->data.op.child_count > 3) { set_error("range requires 2 or 3 arguments"); return NULL; }
    Sequence *start_seq = eval_node(node->data.op.children[0]);
    Sequence *end_seq = eval_node(node->data.op.children[1]);
    if (!start_seq || !end_seq) { seq_free(start_seq); seq_free(end_seq); return NULL; }
    if (start_seq->length != 1 || end_seq->length != 1) { set_error("range: start and end must be single numbers"); seq_free(start_seq); seq_free(end_seq); return NULL; }
    int start = start_seq->values[0];
    int end = end_seq->values[0];
    int step = 1;
    seq_free(start_seq); seq_free(end_seq);
    if (node->data.op.child_count == 3) {
        Sequence *step_seq = eval_node(node->data.op.children[2]);
        if (!step_seq || step_seq->length != 1 || step_seq->values[0] == 0) { set_error("range: step must be non-zero number"); seq_free(step_seq); return NULL; }
        step = step_seq->values[0];
        seq_free(step_seq);
    }
    Sequence *result = seq_new();
    if (!result) return NULL;
    if (step > 0) {
        for (int i = start; i <= end; i += step) {
            if (!seq_append(result, i)) { seq_free(result); return NULL; }
        }
    } else {
        for (int i = start; i >= end; i += step) {
            if (!seq_append(result, i)) { seq_free(result); return NULL; }
        }
    }
    return result;
}

static Sequence* eval_ramp(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("ramp requires 3 arguments"); return NULL; }
    Sequence *start_seq = eval_node(node->data.op.children[0]);
    Sequence *end_seq = eval_node(node->data.op.children[1]);
    Sequence *steps_seq = eval_node(node->data.op.children[2]);
    if (!start_seq || !end_seq || !steps_seq) { seq_free(start_seq); seq_free(end_seq); seq_free(steps_seq); return NULL; }
    if (start_seq->length != 1 || end_seq->length != 1 || steps_seq->length != 1) { set_error("ramp: all arguments must be single numbers"); seq_free(start_seq); seq_free(end_seq); seq_free(steps_seq); return NULL; }
    int start = start_seq->values[0];
    int end = end_seq->values[0];
    int steps = steps_seq->values[0];
    seq_free(start_seq); seq_free(end_seq); seq_free(steps_seq);
    Sequence *result = seq_new();
    if (!result) return NULL;
    if (steps <= 1) {
        if (!seq_append(result, end)) { seq_free(result); return NULL; }
        return result;
    }
    for (int i = 0; i < steps; i++) {
        double t = (double)i / (steps - 1);
        int val = start + (int)(t * (end - start));
        if (!seq_append(result, val)) { seq_free(result); return NULL; }
    }
    return result;
}

static Sequence* eval_drunk(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("drunk requires 3 arguments"); return NULL; }
    Sequence *steps_seq = eval_node(node->data.op.children[0]);
    Sequence *max_step_seq = eval_node(node->data.op.children[1]);
    Sequence *start_seq = eval_node(node->data.op.children[2]);
    if (!steps_seq || !max_step_seq || !start_seq) { seq_free(steps_seq); seq_free(max_step_seq); seq_free(start_seq); return NULL; }
    if (steps_seq->length != 1 || max_step_seq->length != 1 || start_seq->length != 1) { set_error("drunk: all arguments must be single numbers"); seq_free(steps_seq); seq_free(max_step_seq); seq_free(start_seq); return NULL; }
    int steps = steps_seq->values[0];
    int max_step = max_step_seq->values[0];
    int current = start_seq->values[0];
    seq_free(steps_seq); seq_free(max_step_seq); seq_free(start_seq);
    Sequence *result = seq_new();
    if (!result) return NULL;
    for (int i = 0; i < steps; i++) {
        if (!seq_append(result, current)) { seq_free(result); return NULL; }
        int delta = random_range(-max_step, max_step);
        current += delta;
    }
    return result;
}

static Sequence* eval_cycle(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("cycle requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *len_seq = eval_node(node->data.op.children[1]);
    if (!seq || !len_seq) { seq_free(seq); seq_free(len_seq); return NULL; }
    if (len_seq->length != 1 || len_seq->values[0] <= 0) { set_error("cycle: length must be positive number"); seq_free(seq); seq_free(len_seq); return NULL; }
    int target_len = len_seq->values[0];
    seq_free(len_seq);
    if (seq->length == 0) {
        seq_free(seq);
        return seq_new();
    }
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < target_len; i++) {
        if (!seq_append(result, seq->values[i % seq->length])) { seq_free(result); seq_free(seq); return NULL; }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_grow(ASTNode *node) {
    if (node->data.op.child_count != 1) { set_error("grow requires 1 argument"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    if (!seq) return NULL;
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int len = 1; len <= seq->length; len++) {
        for (int i = 0; i < len; i++) {
            if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        }
        for (int i = len; i < seq->length; i++) {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_degrade(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("degrade requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *prob_seq = eval_node(node->data.op.children[1]);
    if (!seq || !prob_seq) { seq_free(seq); seq_free(prob_seq); return NULL; }
    if (prob_seq->length != 1) { set_error("degrade: probability must be single number (0-100)"); seq_free(seq); seq_free(prob_seq); return NULL; }
    int prob = prob_seq->values[0];
    seq_free(prob_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        // Use 0-99 range (100 values) so prob=100 always degrades, prob=0 never degrades
        if (random_range(0, 99) >= prob) {
            if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_transpose(ASTNode *node) {
    return eval_add(node);
}

static Sequence* eval_quantize(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("quantize requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *scale = eval_node(node->data.op.children[1]);
    if (!seq || !scale) { seq_free(seq); seq_free(scale); return NULL; }
    if (scale->length == 0) { set_error("quantize: scale cannot be empty"); seq_free(seq); seq_free(scale); return NULL; }
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); seq_free(scale); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == -1) {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); seq_free(scale); return NULL; }
            continue;
        }
        int nearest = scale->values[0];
        int min_dist = abs(seq->values[i] - nearest);
        for (int j = 1; j < scale->length; j++) {
            int dist = abs(seq->values[i] - scale->values[j]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = scale->values[j];
            }
        }
        if (!seq_append(result, nearest)) { seq_free(result); seq_free(seq); seq_free(scale); return NULL; }
    }
    seq_free(seq); seq_free(scale);
    return result;
}

static Sequence* eval_chord(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("chord requires 2 arguments"); return NULL; }
    Sequence *root_seq = eval_node(node->data.op.children[0]);
    if (!root_seq || root_seq->length != 1) { set_error("chord: root must be single number"); seq_free(root_seq); return NULL; }
    int root = root_seq->values[0];
    seq_free(root_seq);
    Sequence *type_seq = eval_node(node->data.op.children[1]);
    if (!type_seq || type_seq->length != 1) { set_error("chord: type must be single number (0=major, 1=minor)"); seq_free(type_seq); return NULL; }
    int type = type_seq->values[0];
    seq_free(type_seq);
    Sequence *result = seq_new();
    if (!result) return NULL;
    switch (type) {
        case 0: if (!seq_append(result, root) || !seq_append(result, root + 4) || !seq_append(result, root + 7)) { seq_free(result); return NULL; } break;
        case 1: if (!seq_append(result, root) || !seq_append(result, root + 3) || !seq_append(result, root + 7)) { seq_free(result); return NULL; } break;
        case 2: if (!seq_append(result, root) || !seq_append(result, root + 3) || !seq_append(result, root + 6)) { seq_free(result); return NULL; } break;
        case 3: if (!seq_append(result, root) || !seq_append(result, root + 4) || !seq_append(result, root + 8)) { seq_free(result); return NULL; } break;
        case 4: if (!seq_append(result, root) || !seq_append(result, root + 4) || !seq_append(result, root + 7) || !seq_append(result, root + 11)) { seq_free(result); return NULL; } break;
        case 5: if (!seq_append(result, root) || !seq_append(result, root + 3) || !seq_append(result, root + 7) || !seq_append(result, root + 10)) { seq_free(result); return NULL; } break;
        case 6: if (!seq_append(result, root) || !seq_append(result, root + 4) || !seq_append(result, root + 7) || !seq_append(result, root + 10)) { seq_free(result); return NULL; } break;
        default: if (!seq_append(result, root) || !seq_append(result, root + 4) || !seq_append(result, root + 7)) { seq_free(result); return NULL; }
    }
    return result;
}

static Sequence* eval_arp(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("arp requires 3 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *dir_seq = eval_node(node->data.op.children[1]);
    Sequence *len_seq = eval_node(node->data.op.children[2]);
    if (!seq || !dir_seq || !len_seq) { seq_free(seq); seq_free(dir_seq); seq_free(len_seq); return NULL; }
    if (dir_seq->length != 1 || len_seq->length != 1) { set_error("arp: direction and length must be single numbers"); seq_free(seq); seq_free(dir_seq); seq_free(len_seq); return NULL; }
    int direction = dir_seq->values[0];
    int length = len_seq->values[0];
    seq_free(dir_seq); seq_free(len_seq);
    if (seq->length == 0) {
        seq_free(seq);
        return seq_new();
    }
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < length; i++) {
        if (direction == 0) {
            if (!seq_append(result, seq->values[i % seq->length])) { seq_free(result); seq_free(seq); return NULL; }
        } else if (direction == 1) {
            int idx = seq->length - 1 - (i % seq->length);
            if (!seq_append(result, seq->values[idx])) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            int cycle_len = seq->length <= 1 ? 1 : (seq->length - 1) * 2;
            int pos = i % cycle_len;
            int idx = pos < seq->length ? pos : cycle_len - pos;
            if (!seq_append(result, seq->values[idx])) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_delay(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("delay requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1 || n_seq->values[0] < 0) { set_error("delay: n must be non-negative number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < n; i++) {
        if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
    }
    if (!seq_extend(result, seq)) { seq_free(result); seq_free(seq); return NULL; }
    seq_free(seq);
    return result;
}

static Sequence* eval_gate(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("gate requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *n_seq = eval_node(node->data.op.children[1]);
    if (!seq || !n_seq) { seq_free(seq); seq_free(n_seq); return NULL; }
    if (n_seq->length != 1 || n_seq->values[0] <= 0) { set_error("gate: n must be positive number"); seq_free(seq); seq_free(n_seq); return NULL; }
    int n = n_seq->values[0];
    seq_free(n_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (i % n == 0) {
            if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            if (!seq_append(result, -1)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

// ============================================================================
// EVALUATOR DISPATCHER
// ============================================================================

static Sequence* eval_node(ASTNode *node) {
    if (!node) return NULL;
    switch (node->type) {
        case NODE_NUMBER: { Sequence *seq = seq_new(); if (seq && !seq_append(seq, node->data.number)) { seq_free(seq); return NULL; } return seq; }
        case NODE_HYPHEN: { Sequence *seq = seq_new(); if (seq && !seq_append(seq, -1)) { seq_free(seq); return NULL; } return seq; }
        case NODE_SEQ: return eval_seq(node);
        case NODE_REP: return eval_rep(node);
        case NODE_ADD: return eval_add(node);
        case NODE_MUL: return eval_mul(node);
        case NODE_MOD: return eval_mod(node);
        case NODE_SCALE: return eval_scale(node);
        case NODE_CLAMP: return eval_clamp(node);
        case NODE_EUCLID: return eval_euclid(node);
        case NODE_BJORK: return eval_bjork(node);
        case NODE_SUBDIV: return eval_subdiv(node);
        case NODE_REVERSE: return eval_reverse(node);
        case NODE_ROTATE: return eval_rotate(node);
        case NODE_PALINDROME: return eval_palindrome(node);
        case NODE_MIRROR: return eval_mirror(node);
        case NODE_INTERLEAVE: return eval_interleave(node);
        case NODE_SHUFFLE: return eval_shuffle(node);
        case NODE_TAKE: return eval_take(node);
        case NODE_DROP: return eval_drop(node);
        case NODE_EVERY: return eval_every(node);
        case NODE_SLICE: return eval_slice(node);
        case NODE_FILTER: return eval_filter(node);
        case NODE_CHOOSE: return eval_choose(node);
        case NODE_RAND: return eval_rand(node);
        case NODE_PROB: return eval_prob(node);
        case NODE_MAYBE: return eval_maybe(node);
        case NODE_RANGE: return eval_range(node);
        case NODE_RAMP: return eval_ramp(node);
        case NODE_DRUNK: return eval_drunk(node);
        case NODE_CYCLE: return eval_cycle(node);
        case NODE_GROW: return eval_grow(node);
        case NODE_DEGRADE: return eval_degrade(node);
        case NODE_TRANSPOSE: return eval_transpose(node);
        case NODE_QUANTIZE: return eval_quantize(node);
        case NODE_CHORD: return eval_chord(node);
        case NODE_ARP: return eval_arp(node);
        case NODE_DELAY: return eval_delay(node);
        case NODE_GATE: return eval_gate(node);
        default: set_error("Unknown node type"); return NULL;
    }
}

#endif // ALIEN_CORE_H
