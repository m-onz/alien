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
#define ALIEN_VERSION_MINOR 5
#define ALIEN_VERSION_PATCH 0
#define ALIEN_VERSION_STRING "0.5.0"

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

// Rest sentinel — must not collide with valid musical values
#define ALIEN_REST INT_MIN

// Safety limits
#define ALIEN_MAX_SEQ_LEN 65536
#define ALIEN_MAX_DEPTH 64

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
    NODE_SUB,
    NODE_MUL,
    NODE_MOD,
    NODE_SCALE,
    NODE_CLAMP,
    // Rhythm
    NODE_EUCLID,
    NODE_SUBDIV,
    // List manipulation
    NODE_REVERSE,
    NODE_ROTATE,
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
    // Pattern generation
    NODE_RANGE,
    NODE_RAMP,
    NODE_DRUNK,
    // Constraints
    NODE_WRAP,
    NODE_FOLD,
    // Conditional/logic
    NODE_CYCLE,
    NODE_GROW,
    // Musical
    NODE_QUANTIZE,
    NODE_ARP,
    // Time/phase
    NODE_GATE,
    NODE_SPEED,
    NODE_MASK,
    NODE_MIRROR,
    NODE_DELAY
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
    int *values;      // ALIEN_REST represents hyphen/rest
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
static int g_parse_depth = 0;
static int g_eval_depth = 0;

static void init_random(void) {
    if (!g_random_initialized) {
        unsigned int seed = (unsigned int)time(NULL);
        seed ^= (unsigned int)clock();
        seed ^= (unsigned int)(unsigned long)&seed;
        srand(seed);
        g_random_initialized = true;
    }
}

static int random_range(int min, int max) {
    init_random();
    if (max < min) { int t = min; min = max; max = t; }
    long range = (long)max - (long)min + 1;
    if (range <= 0) return min;  // overflow case or degenerate
    return min + (int)(rand() % (int)range);
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
    if (seq->length >= ALIEN_MAX_SEQ_LEN) {
        set_error("Sequence too long");
        return false;
    }
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

// ============================================================================
// SCALE QUANTIZATION HELPER (shared between DSL and PD objects)
// ============================================================================

static int alien_snap_to_scale(int note, const int *scale_pcs, int scale_len, int root) {
    if (scale_len <= 0) return note;
    int octave = (note - root) / 12;
    int pc = (note - root) % 12;
    if (pc < 0) { pc += 12; octave--; }
    int nearest_pc = scale_pcs[0];
    int min_dist = pc - nearest_pc;
    if (min_dist < 0) min_dist = -min_dist;
    if (min_dist > 6) min_dist = 12 - min_dist;
    for (int j = 1; j < scale_len; j++) {
        int d = pc - scale_pcs[j];
        if (d < 0) d = -d;
        if (d > 6) d = 12 - d;
        if (d < min_dist) {
            min_dist = d;
            nearest_pc = scale_pcs[j];
        }
    }
    return octave * 12 + root + nearest_pc;
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
        } else if (*p == '.' || *p == '_') {
            // Alternative rest symbols: . _
            if (*(p+1) == '\0' || isspace(*(p+1)) || *(p+1) == ')' || *(p+1) == '(') {
                tok->type = TOK_HYPHEN;
                p++; column++;
            } else {
                // It's part of a symbol, fall through
                goto parse_symbol;
            }
        } else if (isalpha(*p) || *p == '.' || *p == '_') {
            parse_symbol:
            tok->type = TOK_SYMBOL;
            int i = 0;
            while ((isalpha(*p) || isdigit(*p) || *p == '#' || *p == '.' || *p == '_') && i < ALIEN_MAX_SYMBOL_LEN - 1) {
                tok->value.symbol[i++] = *p++;
                column++;
            }
            tok->value.symbol[i] = '\0';
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
    } else {
        set_error("Too many tokens");
        return -1;
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
    if (++g_parse_depth > ALIEN_MAX_DEPTH) {
        set_error("Maximum nesting depth exceeded");
        return NULL;
    }
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
    else if (strcmp(op, "sub") == 0) node = ast_new_op(NODE_SUB);
    else if (strcmp(op, "mul") == 0) node = ast_new_op(NODE_MUL);
    else if (strcmp(op, "mod") == 0) node = ast_new_op(NODE_MOD);
    else if (strcmp(op, "scale") == 0) node = ast_new_op(NODE_SCALE);
    else if (strcmp(op, "clamp") == 0) node = ast_new_op(NODE_CLAMP);
    else if (strcmp(op, "euclid") == 0) node = ast_new_op(NODE_EUCLID);
    else if (strcmp(op, "subdiv") == 0) node = ast_new_op(NODE_SUBDIV);
    else if (strcmp(op, "reverse") == 0) node = ast_new_op(NODE_REVERSE);
    else if (strcmp(op, "rotate") == 0) node = ast_new_op(NODE_ROTATE);
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
    else if (strcmp(op, "range") == 0) node = ast_new_op(NODE_RANGE);
    else if (strcmp(op, "ramp") == 0) node = ast_new_op(NODE_RAMP);
    else if (strcmp(op, "drunk") == 0) node = ast_new_op(NODE_DRUNK);
    else if (strcmp(op, "wrap") == 0) node = ast_new_op(NODE_WRAP);
    else if (strcmp(op, "fold") == 0) node = ast_new_op(NODE_FOLD);
    else if (strcmp(op, "cycle") == 0) node = ast_new_op(NODE_CYCLE);
    else if (strcmp(op, "grow") == 0) node = ast_new_op(NODE_GROW);
    else if (strcmp(op, "quantize") == 0) node = ast_new_op(NODE_QUANTIZE);
    else if (strcmp(op, "arp") == 0) node = ast_new_op(NODE_ARP);
    else if (strcmp(op, "gate") == 0) node = ast_new_op(NODE_GATE);
    else if (strcmp(op, "speed") == 0) node = ast_new_op(NODE_SPEED);
    else if (strcmp(op, "mask") == 0) node = ast_new_op(NODE_MASK);
    else if (strcmp(op, "mirror") == 0) node = ast_new_op(NODE_MIRROR);
    else if (strcmp(op, "delay") == 0) node = ast_new_op(NODE_DELAY);
    else {
        set_error("Unknown operator");
        return NULL;
    }

    if (!node) {
        set_error("Memory allocation failed");
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
    g_parse_depth--;
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
    g_parse_depth = 0;
    Parser parser = { tokens, 0, token_count };
    ASTNode *ast = parse_expr(&parser);
    if (ast && parser_current(&parser)->type != TOK_EOF) {
        set_error("Unexpected tokens after expression");
        ast_free(ast);
        return NULL;
    }
    return ast;
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
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            long val = (long)seq->values[i] + delta;
            if (val > INT_MAX) val = INT_MAX;
            if (val < INT_MIN + 1) val = INT_MIN + 1;
            if (!seq_append(result, (int)val)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_sub(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("sub requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *delta_seq = eval_node(node->data.op.children[1]);
    if (!seq || !delta_seq) { seq_free(seq); seq_free(delta_seq); return NULL; }
    if (delta_seq->length != 1) { set_error("sub: second arg must be single number"); seq_free(seq); seq_free(delta_seq); return NULL; }
    int delta = delta_seq->values[0];
    seq_free(delta_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            long val = (long)seq->values[i] - delta;
            if (val > INT_MAX) val = INT_MAX;
            if (val < INT_MIN + 1) val = INT_MIN + 1;
            if (!seq_append(result, (int)val)) { seq_free(result); seq_free(seq); return NULL; }
        }
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
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            long val = (long)seq->values[i] * factor;
            if (val > INT_MAX) val = INT_MAX;
            if (val < INT_MIN + 1) val = INT_MIN + 1;
            if (!seq_append(result, (int)val)) { seq_free(result); seq_free(seq); return NULL; }
        }
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
        if (!seq_append(result, seq->values[i] == ALIEN_REST ? ALIEN_REST : seq->values[i] % divisor)) { seq_free(result); seq_free(seq); return NULL; }
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
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
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
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
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
    // euclid hits steps [rotation [hit_value]]
    if (node->data.op.child_count < 2 || node->data.op.child_count > 4) { set_error("euclid requires 2-4 arguments: hits steps [rotation [hit_value]]"); return NULL; }

    Sequence *pattern_seq = eval_node(node->data.op.children[0]);
    Sequence *steps_seq = eval_node(node->data.op.children[1]);

    if (!pattern_seq || !steps_seq) { seq_free(pattern_seq); seq_free(steps_seq); return NULL; }
    if (steps_seq->length != 1) { set_error("euclid: steps must be single number"); seq_free(pattern_seq); seq_free(steps_seq); return NULL; }

    int steps = steps_seq->values[0];
    seq_free(steps_seq);

    if (steps <= 0) { set_error("euclid: steps must be positive"); seq_free(pattern_seq); return NULL; }

    // Handle zero hits explicitly
    if (pattern_seq->length == 1 && pattern_seq->values[0] == 0) {
        seq_free(pattern_seq);
        Sequence *result = seq_new();
        if (!result) return NULL;
        for (int i = 0; i < steps; i++) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); return NULL; }
        }
        return result;
    }

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
    if (node->data.op.child_count >= 3) {
        Sequence *rot_seq = eval_node(node->data.op.children[2]);
        if (!rot_seq || rot_seq->length != 1) { set_error("euclid: rotation must be single number"); seq_free(rot_seq); seq_free(pattern_seq); return NULL; }
        rotation = rot_seq->values[0];
        seq_free(rot_seq);
    }

    // Optional hit value (4th arg) — overrides the default 1 for numeric hit counts
    if (node->data.op.child_count == 4 && is_hit_count) {
        Sequence *val_seq = eval_node(node->data.op.children[3]);
        if (!val_seq || val_seq->length != 1) { set_error("euclid: hit_value must be single number"); seq_free(val_seq); seq_free(pattern_seq); return NULL; }
        seq_free(pattern_seq);
        pattern_seq = seq_new();
        if (!pattern_seq) { seq_free(val_seq); return NULL; }
        if (!seq_append(pattern_seq, val_seq->values[0])) { seq_free(pattern_seq); seq_free(val_seq); return NULL; }
        seq_free(val_seq);
    }

    int *euclid_pattern = (int*)ALIEN_MALLOC(sizeof(int) * steps);
    if (!euclid_pattern) { seq_free(pattern_seq); return NULL; }
    euclidean_rhythm(hits, steps, euclid_pattern);

    Sequence *result = seq_new();
    if (!result) { ALIEN_FREE(euclid_pattern, sizeof(int) * steps); seq_free(pattern_seq); return NULL; }
    int pattern_idx = 0;
    for (int i = 0; i < steps; i++) {
        int idx = ((i + rotation) % steps + steps) % steps;  // safe modulo for negative rotation
        if (euclid_pattern[idx]) {
            if (!seq_append(result, pattern_seq->values[pattern_idx % pattern_seq->length])) { seq_free(result); ALIEN_FREE(euclid_pattern, sizeof(int) * steps); seq_free(pattern_seq); return NULL; }
            pattern_idx++;
        } else {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); ALIEN_FREE(euclid_pattern, sizeof(int) * steps); seq_free(pattern_seq); return NULL; }
        }
    }

    ALIEN_FREE(euclid_pattern, sizeof(int) * steps);
    seq_free(pattern_seq);
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
        if (seq->values[i] != ALIEN_REST) {
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
    // rand count [min max] - defaults to MIDI range 0-127
    if (node->data.op.child_count != 1 && node->data.op.child_count != 3) {
        set_error("rand requires 1 or 3 arguments: count [min max]");
        return NULL;
    }
    Sequence *count_seq = eval_node(node->data.op.children[0]);
    if (!count_seq || count_seq->length != 1) { 
        set_error("rand: count must be single number"); 
        seq_free(count_seq); 
        return NULL; 
    }
    int count = count_seq->values[0];
    seq_free(count_seq);
    
    // Default to MIDI range
    int min = 0, max = 127;
    
    if (node->data.op.child_count >= 3) {
        Sequence *min_seq = eval_node(node->data.op.children[1]);
        Sequence *max_seq = eval_node(node->data.op.children[2]);
        if (!min_seq || !max_seq || min_seq->length != 1 || max_seq->length != 1) { 
            set_error("rand: min and max must be single numbers"); 
            seq_free(min_seq); seq_free(max_seq); 
            return NULL; 
        }
        min = min_seq->values[0];
        max = max_seq->values[0];
        seq_free(min_seq); seq_free(max_seq);
    }
    
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
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
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
        int val = start + (int)round(t * (end - start));
        if (!seq_append(result, val)) { seq_free(result); return NULL; }
    }
    return result;
}

static Sequence* eval_drunk(ASTNode *node) {
    // drunk steps max_step start [min max]
    if (node->data.op.child_count < 3 || node->data.op.child_count > 5) { 
        set_error("drunk requires 3-5 arguments: steps max_step start [min max]"); 
        return NULL; 
    }
    Sequence *steps_seq = eval_node(node->data.op.children[0]);
    Sequence *max_step_seq = eval_node(node->data.op.children[1]);
    Sequence *start_seq = eval_node(node->data.op.children[2]);
    if (!steps_seq || !max_step_seq || !start_seq) { seq_free(steps_seq); seq_free(max_step_seq); seq_free(start_seq); return NULL; }
    if (steps_seq->length != 1 || max_step_seq->length != 1 || start_seq->length != 1) { set_error("drunk: steps, max_step, start must be single numbers"); seq_free(steps_seq); seq_free(max_step_seq); seq_free(start_seq); return NULL; }
    int steps = steps_seq->values[0];
    int max_step = max_step_seq->values[0];
    int current = start_seq->values[0];
    seq_free(steps_seq); seq_free(max_step_seq); seq_free(start_seq);
    
    // Optional bounds
    int has_bounds = 0;
    int min_val = 0, max_val = 127;
    if (node->data.op.child_count >= 5) {
        Sequence *min_seq = eval_node(node->data.op.children[3]);
        Sequence *max_seq = eval_node(node->data.op.children[4]);
        if (!min_seq || !max_seq || min_seq->length != 1 || max_seq->length != 1) {
            set_error("drunk: min and max must be single numbers");
            seq_free(min_seq); seq_free(max_seq);
            return NULL;
        }
        min_val = min_seq->values[0];
        max_val = max_seq->values[0];
        seq_free(min_seq); seq_free(max_seq);
        if (max_val < min_val) { set_error("drunk: max must be >= min"); return NULL; }
        has_bounds = 1;
    }

    Sequence *result = seq_new();
    if (!result) return NULL;
    for (int i = 0; i < steps; i++) {
        if (!seq_append(result, current)) { seq_free(result); return NULL; }
        int delta = random_range(-max_step, max_step);
        current += delta;
        // Apply bounds if specified (clamp — safe for any range including min==max)
        if (has_bounds) {
            if (min_val == max_val) {
                current = min_val;
            } else {
                // Reflect at boundaries with iteration limit to prevent infinite loop
                int limit = 4;
                while ((current < min_val || current > max_val) && limit-- > 0) {
                    if (current < min_val) current = min_val + (min_val - current);
                    if (current > max_val) current = max_val - (current - max_val);
                }
                // Final clamp as safety net
                if (current < min_val) current = min_val;
                if (current > max_val) current = max_val;
            }
        }
    }
    return result;
}

// wrap: constrain values to range using modulo (octave wrapping)
static Sequence* eval_wrap(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("wrap requires 3 arguments: seq min max"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *min_seq = eval_node(node->data.op.children[1]);
    Sequence *max_seq = eval_node(node->data.op.children[2]);
    if (!seq || !min_seq || !max_seq) { seq_free(seq); seq_free(min_seq); seq_free(max_seq); return NULL; }
    if (min_seq->length != 1 || max_seq->length != 1) { 
        set_error("wrap: min and max must be single numbers"); 
        seq_free(seq); seq_free(min_seq); seq_free(max_seq); 
        return NULL; 
    }
    int min_val = min_seq->values[0];
    int max_val = max_seq->values[0];
    seq_free(min_seq); seq_free(max_seq);
    
    if (max_val <= min_val) { set_error("wrap: max must be greater than min"); seq_free(seq); return NULL; }
    int range = max_val - min_val;
    
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            int val = seq->values[i];
            // Wrap to range using modulo
            val = ((val - min_val) % range + range) % range + min_val;
            if (!seq_append(result, val)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

// fold: constrain values to range by reflecting at boundaries
static Sequence* eval_fold(ASTNode *node) {
    if (node->data.op.child_count != 3) { set_error("fold requires 3 arguments: seq min max"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *min_seq = eval_node(node->data.op.children[1]);
    Sequence *max_seq = eval_node(node->data.op.children[2]);
    if (!seq || !min_seq || !max_seq) { seq_free(seq); seq_free(min_seq); seq_free(max_seq); return NULL; }
    if (min_seq->length != 1 || max_seq->length != 1) { 
        set_error("fold: min and max must be single numbers"); 
        seq_free(seq); seq_free(min_seq); seq_free(max_seq); 
        return NULL; 
    }
    int min_val = min_seq->values[0];
    int max_val = max_seq->values[0];
    seq_free(min_seq); seq_free(max_seq);
    
    if (max_val <= min_val) { set_error("fold: max must be greater than min"); seq_free(seq); return NULL; }

    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        } else {
            int val = seq->values[i];
            // Fold/reflect at boundaries with iteration limit
            int limit = 4;
            while ((val < min_val || val > max_val) && limit-- > 0) {
                if (val < min_val) val = min_val + (min_val - val);
                if (val > max_val) val = max_val - (val - max_val);
            }
            // Final clamp as safety net
            if (val < min_val) val = min_val;
            if (val > max_val) val = max_val;
            if (!seq_append(result, val)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
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
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_quantize(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("quantize requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *scale = eval_node(node->data.op.children[1]);
    if (!seq || !scale) { seq_free(seq); seq_free(scale); return NULL; }
    if (scale->length == 0) { set_error("quantize: scale cannot be empty"); seq_free(seq); seq_free(scale); return NULL; }
    // Detect pitch-class scale (all values 0-11) for octave-aware quantization
    int is_pitch_class = 1;
    for (int j = 0; j < scale->length; j++) {
        if (scale->values[j] < 0 || scale->values[j] > 11) {
            is_pitch_class = 0;
            break;
        }
    }

    Sequence *result = seq_new();
    if (!result) { seq_free(seq); seq_free(scale); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (seq->values[i] == ALIEN_REST) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); seq_free(scale); return NULL; }
            continue;
        }
        int quantized;
        if (is_pitch_class) {
            // Octave-aware: match pitch class, preserve octave
            int note = seq->values[i];
            int octave = note / 12;
            int pc = note % 12;
            if (pc < 0) { pc += 12; octave--; }
            int nearest_pc = scale->values[0];
            int min_dist = abs(pc - nearest_pc);
            if (min_dist > 6) min_dist = 12 - min_dist;
            for (int j = 1; j < scale->length; j++) {
                int d = abs(pc - scale->values[j]);
                if (d > 6) d = 12 - d;
                if (d < min_dist) {
                    min_dist = d;
                    nearest_pc = scale->values[j];
                }
            }
            quantized = octave * 12 + nearest_pc;
        } else {
            // Absolute matching
            quantized = scale->values[0];
            int min_dist = abs(seq->values[i] - quantized);
            for (int j = 1; j < scale->length; j++) {
                int dist = abs(seq->values[i] - scale->values[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    quantized = scale->values[j];
                }
            }
        }
        if (!seq_append(result, quantized)) { seq_free(result); seq_free(seq); seq_free(scale); return NULL; }
    }
    seq_free(seq); seq_free(scale);
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
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_speed(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("speed requires 2 arguments"); return NULL; }
    Sequence *seq = eval_node(node->data.op.children[0]);
    Sequence *div_seq = eval_node(node->data.op.children[1]);
    if (!seq || !div_seq) { seq_free(seq); seq_free(div_seq); return NULL; }
    if (div_seq->length != 1 || div_seq->values[0] <= 0) { set_error("speed: division must be positive number"); seq_free(seq); seq_free(div_seq); return NULL; }
    int div = div_seq->values[0];
    seq_free(div_seq);
    Sequence *result = seq_new();
    if (!result) { seq_free(seq); return NULL; }
    for (int i = 0; i < seq->length; i++) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
        for (int j = 1; j < div; j++) {
            if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
        }
    }
    seq_free(seq);
    return result;
}

static Sequence* eval_mask(ASTNode *node) {
    if (node->data.op.child_count != 2) { set_error("mask requires 2 arguments"); return NULL; }
    Sequence *source = eval_node(node->data.op.children[0]);
    Sequence *gate = eval_node(node->data.op.children[1]);
    if (!source || !gate) { seq_free(source); seq_free(gate); return NULL; }
    if (source->length == 0) { seq_free(source); seq_free(gate); return seq_new(); }
    Sequence *result = seq_new();
    if (!result) { seq_free(source); seq_free(gate); return NULL; }
    int src_idx = 0;
    for (int i = 0; i < gate->length; i++) {
        if (gate->values[i] != ALIEN_REST) {
            if (!seq_append(result, source->values[src_idx % source->length])) {
                seq_free(result); seq_free(source); seq_free(gate); return NULL;
            }
            src_idx++;
        } else {
            if (!seq_append(result, ALIEN_REST)) {
                seq_free(result); seq_free(source); seq_free(gate); return NULL;
            }
        }
    }
    seq_free(source); seq_free(gate);
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
    for (int i = seq->length - 2; i >= 0; i--) {
        if (!seq_append(result, seq->values[i])) { seq_free(result); seq_free(seq); return NULL; }
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
        if (!seq_append(result, ALIEN_REST)) { seq_free(result); seq_free(seq); return NULL; }
    }
    if (!seq_extend(result, seq)) { seq_free(result); seq_free(seq); return NULL; }
    seq_free(seq);
    return result;
}

// ============================================================================
// EVALUATOR DISPATCHER
// ============================================================================

static Sequence* eval_node(ASTNode *node) {
    if (!node) return NULL;
    if (++g_eval_depth > ALIEN_MAX_DEPTH) {
        set_error("Maximum evaluation depth exceeded");
        g_eval_depth--;
        return NULL;
    }
    Sequence *result = NULL;
    switch (node->type) {
        case NODE_NUMBER: {
            result = seq_new();
            if (result && !seq_append(result, node->data.number)) { seq_free(result); result = NULL; }
            break;
        }
        case NODE_HYPHEN: {
            result = seq_new();
            if (result && !seq_append(result, ALIEN_REST)) { seq_free(result); result = NULL; }
            break;
        }
        case NODE_SEQ: result = eval_seq(node); break;
        case NODE_REP: result = eval_rep(node); break;
        case NODE_ADD: result = eval_add(node); break;
        case NODE_SUB: result = eval_sub(node); break;
        case NODE_MUL: result = eval_mul(node); break;
        case NODE_MOD: result = eval_mod(node); break;
        case NODE_SCALE: result = eval_scale(node); break;
        case NODE_CLAMP: result = eval_clamp(node); break;
        case NODE_EUCLID: result = eval_euclid(node); break;
        case NODE_SUBDIV: result = eval_subdiv(node); break;
        case NODE_REVERSE: result = eval_reverse(node); break;
        case NODE_ROTATE: result = eval_rotate(node); break;
        case NODE_INTERLEAVE: result = eval_interleave(node); break;
        case NODE_SHUFFLE: result = eval_shuffle(node); break;
        case NODE_TAKE: result = eval_take(node); break;
        case NODE_DROP: result = eval_drop(node); break;
        case NODE_EVERY: result = eval_every(node); break;
        case NODE_SLICE: result = eval_slice(node); break;
        case NODE_FILTER: result = eval_filter(node); break;
        case NODE_CHOOSE: result = eval_choose(node); break;
        case NODE_RAND: result = eval_rand(node); break;
        case NODE_PROB: result = eval_prob(node); break;
        case NODE_RANGE: result = eval_range(node); break;
        case NODE_RAMP: result = eval_ramp(node); break;
        case NODE_DRUNK: result = eval_drunk(node); break;
        case NODE_WRAP: result = eval_wrap(node); break;
        case NODE_FOLD: result = eval_fold(node); break;
        case NODE_CYCLE: result = eval_cycle(node); break;
        case NODE_GROW: result = eval_grow(node); break;
        case NODE_QUANTIZE: result = eval_quantize(node); break;
        case NODE_ARP: result = eval_arp(node); break;
        case NODE_GATE: result = eval_gate(node); break;
        case NODE_SPEED: result = eval_speed(node); break;
        case NODE_MASK: result = eval_mask(node); break;
        case NODE_MIRROR: result = eval_mirror(node); break;
        case NODE_DELAY: result = eval_delay(node); break;
        default: set_error("Unknown node type"); break;
    }
    g_eval_depth--;
    return result;
}

#endif // ALIEN_CORE_H
