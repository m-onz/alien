/*
 * alien_evolve.c - Self-contained pattern evolution external for Pure Data
 *
 * Generates evolved alien DSL patterns using genetic algorithms,
 * n-gram models, and random generation with arity constraints.
 */

#include "m_pd.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "alien_evo_core.h"

// ============================================================================
// CONSTANTS
// ============================================================================

#define DEFAULT_POP_SIZE     50
#define DEFAULT_MAX_DEPTH    4
#define DEFAULT_CORPUS_MAX   1000
#define DEFAULT_MUTATION     30
#define DEFAULT_CROSSOVER    70
#define DEFAULT_ELITE        5
#define DEFAULT_TOURNAMENT   3
#define DEFAULT_DIVERSITY    50

#define MODE_EVOLVE  0
#define MODE_RANDOM  1
#define MODE_NGRAM   2

#define MAX_PATTERN_LEN 1024

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    char *source;
    ASTNode *ast;
    uint32_t hash;
    float fitness;
    int protected;  // Seed patterns are protected
    int age;
} CorpusEntry;

typedef struct {
    ASTNode *genome;
    char *source;
    float fitness;
} Individual;

typedef struct _alien_evolve {
    t_object x_obj;

    // === Corpus ===
    CorpusEntry *corpus;
    int corpus_count;
    int corpus_capacity;
    int corpus_max;

    // === N-gram Model ===
    char *ngram_text;
    uint32_t ngram_len;
    uint32_t *suffix_array;
    int ngram_built;

    // === Population ===
    Individual *population;
    int pop_size;
    int pop_capacity;
    int generation;

    // === Configuration ===
    int max_depth;
    float mutation_rate;
    float crossover_rate;
    int elite_count;
    int tournament_size;
    float diversity_weight;
    int mode;

    // === Outlets ===
    t_outlet *out_pattern;
    t_outlet *out_info;
} t_alien_evolve;

static t_class *alien_evolve_class;

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

static void evolve_free_corpus_entry(CorpusEntry *entry);
static void evolve_free_individual(Individual *ind);
static void evolve_rebuild_ngram(t_alien_evolve *x);
static ASTNode* evolve_random_ast(t_alien_evolve *x, int max_depth);
static float evolve_evaluate_fitness(t_alien_evolve *x, ASTNode *node);

// ============================================================================
// CORPUS MANAGEMENT
// ============================================================================

static void evolve_free_corpus_entry(CorpusEntry *entry) {
    if (entry->source) {
        ALIEN_FREE(entry->source, strlen(entry->source) + 1);
        entry->source = NULL;
    }
    if (entry->ast) {
        ast_free(entry->ast);
        entry->ast = NULL;
    }
}

static int evolve_corpus_add(t_alien_evolve *x, const char *pattern, int is_seed) {
    // Parse the pattern
    Token tokens[256];
    int token_count = tokenize(pattern, tokens, 256);
    if (token_count < 0) return 0;

    ASTNode *ast = parse(tokens, token_count);
    if (!ast) return 0;

    if (!evo_ast_valid(ast)) {
        ast_free(ast);
        return 0;
    }

    uint32_t hash = evo_str_hash(pattern);

    // Check for duplicates
    for (int i = 0; i < x->corpus_count; i++) {
        if (x->corpus[i].hash == hash && strcmp(x->corpus[i].source, pattern) == 0) {
            ast_free(ast);
            return 0;  // Already exists
        }
    }

    // Evict if at capacity (don't evict protected entries)
    while (x->corpus_count >= x->corpus_max) {
        int evict_idx = -1;
        float lowest_fitness = 1e9;
        int oldest_age = -1;

        for (int i = 0; i < x->corpus_count; i++) {
            if (x->corpus[i].protected) continue;

            // Prefer to evict low fitness, old entries
            float score = x->corpus[i].fitness - (x->corpus[i].age * 0.01f);
            if (evict_idx < 0 || score < lowest_fitness ||
                (score == lowest_fitness && x->corpus[i].age > oldest_age)) {
                evict_idx = i;
                lowest_fitness = score;
                oldest_age = x->corpus[i].age;
            }
        }

        if (evict_idx < 0) {
            // All entries are protected
            ast_free(ast);
            return 0;
        }

        evolve_free_corpus_entry(&x->corpus[evict_idx]);
        // Shift remaining entries
        for (int i = evict_idx; i < x->corpus_count - 1; i++) {
            x->corpus[i] = x->corpus[i + 1];
        }
        x->corpus_count--;
    }

    // Expand capacity if needed
    if (x->corpus_count >= x->corpus_capacity) {
        int old_cap = x->corpus_capacity;
        int new_cap = old_cap * 2;
        CorpusEntry *new_corpus = (CorpusEntry*)ALIEN_REALLOC(x->corpus,
            sizeof(CorpusEntry) * old_cap, sizeof(CorpusEntry) * new_cap);
        if (!new_corpus) {
            ast_free(ast);
            return 0;
        }
        x->corpus = new_corpus;
        x->corpus_capacity = new_cap;
    }

    // Add new entry
    CorpusEntry *entry = &x->corpus[x->corpus_count];
    size_t src_len = strlen(pattern) + 1;
    entry->source = (char*)ALIEN_MALLOC(src_len);
    if (!entry->source) {
        ast_free(ast);
        return 0;
    }
    memcpy(entry->source, pattern, src_len);
    entry->ast = ast;
    entry->hash = hash;
    entry->fitness = evolve_evaluate_fitness(x, ast);
    entry->protected = is_seed;
    entry->age = 0;
    x->corpus_count++;

    x->ngram_built = 0;  // Invalidate n-gram
    return 1;
}

static void evolve_corpus_age(t_alien_evolve *x) {
    for (int i = 0; i < x->corpus_count; i++) {
        x->corpus[i].age++;
    }
}

// ============================================================================
// N-GRAM MODEL (SUFFIX ARRAY)
// ============================================================================

static int suffix_cmp(const void *a, const void *b, void *arg) {
    const char *text = (const char*)arg;
    uint32_t ia = *(const uint32_t*)a;
    uint32_t ib = *(const uint32_t*)b;
    return strcmp(text + ia, text + ib);
}

// qsort_r is not portable, so we use a simple implementation
static const char *g_suffix_text = NULL;

static int suffix_cmp_global(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t*)a;
    uint32_t ib = *(const uint32_t*)b;
    return strcmp(g_suffix_text + ia, g_suffix_text + ib);
}

static void evolve_rebuild_ngram(t_alien_evolve *x) {
    if (x->ngram_built) return;

    // Free old data
    if (x->ngram_text) {
        ALIEN_FREE(x->ngram_text, x->ngram_len + 1);
        x->ngram_text = NULL;
    }
    if (x->suffix_array) {
        ALIEN_FREE(x->suffix_array, sizeof(uint32_t) * x->ngram_len);
        x->suffix_array = NULL;
    }

    // Calculate total length
    uint32_t total_len = 0;
    for (int i = 0; i < x->corpus_count; i++) {
        total_len += (uint32_t)strlen(x->corpus[i].source) + 1;  // +1 for newline
    }

    if (total_len == 0) {
        x->ngram_built = 1;
        return;
    }

    // Allocate and concatenate
    x->ngram_text = (char*)ALIEN_MALLOC(total_len + 1);
    if (!x->ngram_text) return;

    char *ptr = x->ngram_text;
    for (int i = 0; i < x->corpus_count; i++) {
        size_t len = strlen(x->corpus[i].source);
        memcpy(ptr, x->corpus[i].source, len);
        ptr += len;
        *ptr++ = '\n';
    }
    *ptr = '\0';
    x->ngram_len = total_len;

    // Build suffix array
    x->suffix_array = (uint32_t*)ALIEN_MALLOC(sizeof(uint32_t) * total_len);
    if (!x->suffix_array) {
        ALIEN_FREE(x->ngram_text, total_len + 1);
        x->ngram_text = NULL;
        return;
    }

    for (uint32_t i = 0; i < total_len; i++) {
        x->suffix_array[i] = i;
    }

    // Sort using global variable (not ideal but portable)
    g_suffix_text = x->ngram_text;
    qsort(x->suffix_array, total_len, sizeof(uint32_t), suffix_cmp_global);
    g_suffix_text = NULL;

    x->ngram_built = 1;
}

// Binary search for suffix array range matching a context
static void evolve_ngram_range(t_alien_evolve *x, const char *context,
                                uint32_t *start, uint32_t *end) {
    if (!x->ngram_text || !x->suffix_array || x->ngram_len == 0) {
        *start = *end = 0;
        return;
    }

    size_t ctx_len = strlen(context);

    // Binary search for lower bound
    uint32_t lo = 0, hi = x->ngram_len;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (strncmp(x->ngram_text + x->suffix_array[mid], context, ctx_len) < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *start = lo;

    // Binary search for upper bound
    hi = x->ngram_len;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (strncmp(x->ngram_text + x->suffix_array[mid], context, ctx_len) <= 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *end = lo;
}

// Sample next character from n-gram model with backoff
static char evolve_sample_next_char(t_alien_evolve *x, const char *context) {
    if (!x->ngram_text || !x->suffix_array) {
        // Fallback: random digit or space
        int r = random_range(0, 11);
        if (r < 10) return '0' + r;
        return ' ';
    }

    // Try progressively shorter contexts (backoff)
    size_t ctx_len = strlen(context);
    for (size_t backoff = 0; backoff < ctx_len && backoff < 20; backoff++) {
        const char *ctx = context + backoff;
        uint32_t start, end;
        evolve_ngram_range(x, ctx, &start, &end);

        if (end > start) {
            // Count character distribution at position ctx_len - backoff
            int counts[128] = {0};
            int total = 0;
            size_t pos = ctx_len - backoff;

            for (uint32_t i = start; i < end && i < start + 1000; i++) {
                uint32_t suffix_pos = x->suffix_array[i] + (uint32_t)pos;
                if (suffix_pos < x->ngram_len) {
                    unsigned char c = (unsigned char)x->ngram_text[suffix_pos];
                    if (c < 128 && c > 0) {
                        counts[c]++;
                        total++;
                    }
                }
            }

            if (total > 0) {
                int r = random_range(0, total - 1);
                int cumulative = 0;
                for (int c = 0; c < 128; c++) {
                    cumulative += counts[c];
                    if (cumulative > r) {
                        return (char)c;
                    }
                }
            }
        }
    }

    // Fallback to uniform distribution over common characters
    const char *common = "0123456789 ()";
    return common[random_range(0, (int)strlen(common) - 1)];
}

// Complete a pattern using n-gram model
static char* evolve_ngram_complete(t_alien_evolve *x, const char *start, int max_len) {
    size_t start_len = strlen(start);
    char *result = (char*)ALIEN_MALLOC(max_len + 1);
    if (!result) return NULL;

    memcpy(result, start, start_len);
    int pos = (int)start_len;

    // Count paren depth
    int depth = 0;
    for (int i = 0; i < pos; i++) {
        if (result[i] == '(') depth++;
        if (result[i] == ')') depth--;
    }

    while (pos < max_len && depth > 0) {
        // Use last N characters as context
        int ctx_start = pos > 20 ? pos - 20 : 0;
        char context[32];
        int ctx_len = pos - ctx_start;
        memcpy(context, result + ctx_start, ctx_len);
        context[ctx_len] = '\0';

        char next = evolve_sample_next_char(x, context);
        if (next == '\n' || next == '\0') break;

        result[pos++] = next;
        if (next == '(') depth++;
        if (next == ')') depth--;
    }

    result[pos] = '\0';
    return result;
}

// ============================================================================
// RANDOM AST GENERATION
// ============================================================================

// Simple non-recursive random AST generation
static ASTNode* evolve_random_ast(t_alien_evolve *x, int max_depth) {
    (void)max_depth;  // Ignore depth for now, keep it simple

    int ptype = random_range(0, 7);
    ASTNode *node = NULL;
    ASTNode *child = NULL;
    int i;

    switch (ptype) {
        case 0:  // (euclid hits steps)
            node = ast_new_op(NODE_EUCLID);
            if (!node) return NULL;
            child = ast_new_number(random_range(2, 7));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            child = ast_new_number(random_range(8, 16));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        case 1:  // (seq n n n n)
            node = ast_new_op(NODE_SEQ);
            if (!node) return NULL;
            for (i = 0; i < random_range(3, 6); i++) {
                child = ast_new_number(random_range(48, 84));
                if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            }
            break;

        case 2:  // (chord root type)
            node = ast_new_op(NODE_CHORD);
            if (!node) return NULL;
            child = ast_new_number(random_range(48, 72));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            child = ast_new_number(random_range(0, 6));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        case 3:  // (range start end)
            node = ast_new_op(NODE_RANGE);
            if (!node) return NULL;
            child = ast_new_number(random_range(48, 60));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            child = ast_new_number(random_range(60, 84));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        case 4:  // (reverse (seq n n n))
            node = ast_new_op(NODE_REVERSE);
            if (!node) return NULL;
            child = ast_new_op(NODE_SEQ);
            if (!child) { ast_free(node); return NULL; }
            for (i = 0; i < random_range(3, 5); i++) {
                ASTNode *num = ast_new_number(random_range(48, 84));
                if (!num || !ast_add_child(child, num)) { ast_free(num); ast_free(child); ast_free(node); return NULL; }
            }
            if (!ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        case 5:  // (rotate (seq n n n) r)
            node = ast_new_op(NODE_ROTATE);
            if (!node) return NULL;
            child = ast_new_op(NODE_SEQ);
            if (!child) { ast_free(node); return NULL; }
            for (i = 0; i < random_range(4, 6); i++) {
                ASTNode *num = ast_new_number(random_range(48, 84));
                if (!num || !ast_add_child(child, num)) { ast_free(num); ast_free(child); ast_free(node); return NULL; }
            }
            if (!ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            child = ast_new_number(random_range(1, 4));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        case 6:  // (palindrome (seq n n n))
            node = ast_new_op(NODE_PALINDROME);
            if (!node) return NULL;
            child = ast_new_op(NODE_SEQ);
            if (!child) { ast_free(node); return NULL; }
            for (i = 0; i < random_range(3, 5); i++) {
                ASTNode *num = ast_new_number(random_range(48, 84));
                if (!num || !ast_add_child(child, num)) { ast_free(num); ast_free(child); ast_free(node); return NULL; }
            }
            if (!ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        case 7:  // (take (range a b) n)
            node = ast_new_op(NODE_TAKE);
            if (!node) return NULL;
            child = ast_new_op(NODE_RANGE);
            if (!child) { ast_free(node); return NULL; }
            {
                ASTNode *num = ast_new_number(random_range(48, 60));
                if (!num || !ast_add_child(child, num)) { ast_free(num); ast_free(child); ast_free(node); return NULL; }
                num = ast_new_number(random_range(60, 84));
                if (!num || !ast_add_child(child, num)) { ast_free(num); ast_free(child); ast_free(node); return NULL; }
            }
            if (!ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            child = ast_new_number(random_range(4, 8));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;

        default:
            // Fallback to simple euclid
            node = ast_new_op(NODE_EUCLID);
            if (!node) return NULL;
            child = ast_new_number(random_range(3, 5));
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            child = ast_new_number(8);
            if (!child || !ast_add_child(node, child)) { ast_free(child); ast_free(node); return NULL; }
            break;
    }

    return node;
}

// ============================================================================
// GENETIC OPERATIONS
// ============================================================================

// Crossover: swap subtrees between parents
static ASTNode* evolve_crossover(t_alien_evolve *x, ASTNode *p1, ASTNode *p2) {
    (void)x;

    // Clone first parent
    ASTNode *child = evo_ast_clone(p1);
    if (!child) return NULL;

    int child_count = evo_ast_node_count(child);
    int p2_count = evo_ast_node_count(p2);

    if (child_count <= 1 || p2_count <= 1) return child;

    // Select random crossover point in child (skip root)
    int target_idx = random_range(1, child_count - 1);
    ASTNode *target = evo_random_subtree(child, &target_idx);

    // Select random subtree from p2
    int donor_idx = random_range(0, p2_count - 1);
    ASTNode *donor = evo_random_subtree((ASTNode*)p2, &donor_idx);

    if (target && donor) {
        ASTNode *donor_clone = evo_ast_clone(donor);
        if (donor_clone) {
            if (!evo_replace_subtree(child, target, donor_clone)) {
                ast_free(donor_clone);
            }
        }
    }

    // Validate result
    if (!evo_ast_valid(child)) {
        ast_free(child);
        return evo_ast_clone(p1);  // Fall back to clone
    }

    return child;
}

// Point mutation: change a single value
static void evolve_mutate_point(t_alien_evolve *x, ASTNode *node) {
    (void)x;

    NodeList *leaves = nodelist_new();
    if (!leaves) return;

    evo_collect_number_leaves(node, leaves);

    if (leaves->count > 0) {
        int idx = random_range(0, leaves->count - 1);
        ASTNode *leaf = leaves->nodes[idx];

        // Mutate value
        int mutation_type = random_range(0, 2);
        if (mutation_type == 0) {
            // Small delta
            int delta = random_range(-5, 5);
            leaf->data.number += delta;
        } else if (mutation_type == 1) {
            // Replace with nearby value
            leaf->data.number = random_range(
                leaf->data.number - 12,
                leaf->data.number + 12
            );
        } else {
            // Random value
            leaf->data.number = random_range(0, 127);
        }

        // Clamp to reasonable range
        if (leaf->data.number < 0) leaf->data.number = 0;
        if (leaf->data.number > 127) leaf->data.number = 127;
    }

    nodelist_free(leaves);
}

// Structural mutation: add, remove, or swap operators
static ASTNode* evolve_mutate_structure(t_alien_evolve *x, ASTNode *node) {
    if (!node) return NULL;

    ASTNode *mutant = evo_ast_clone(node);
    if (!mutant) return NULL;

    int node_count = evo_ast_node_count(mutant);
    if (node_count <= 1) {
        // Wrap in an operator
        int op_idx = random_range(0, (int)OP_META_COUNT - 1);
        const OpMeta *meta = &g_op_meta[op_idx];

        if (meta->min_args == 1) {
            ASTNode *wrapper = ast_new_op(meta->type);
            if (wrapper && ast_add_child(wrapper, mutant)) {
                return wrapper;
            }
            ast_free(wrapper);
        }
        return mutant;
    }

    int mutation = random_range(0, 2);

    if (mutation == 0) {
        // Swap an operator type
        int target_idx = random_range(0, node_count - 1);
        ASTNode *target = evo_random_subtree(mutant, &target_idx);

        if (target && target->type != NODE_NUMBER && target->type != NODE_HYPHEN) {
            // Find compatible operator (same arity range)
            int argc = target->data.op.child_count;
            for (int tries = 0; tries < 10; tries++) {
                int op_idx = random_range(0, (int)OP_META_COUNT - 1);
                const OpMeta *meta = &g_op_meta[op_idx];
                if (argc >= meta->min_args &&
                    (meta->max_args < 0 || argc <= meta->max_args)) {
                    target->type = meta->type;
                    break;
                }
            }
        }
    } else if (mutation == 1) {
        // Insert wrapper around subtree
        int target_idx = random_range(1, node_count - 1);
        ASTNode *target = evo_random_subtree(mutant, &target_idx);

        if (target) {
            // Find single-arg operator
            for (int tries = 0; tries < 10; tries++) {
                int op_idx = random_range(0, (int)OP_META_COUNT - 1);
                const OpMeta *meta = &g_op_meta[op_idx];
                if (meta->min_args == 1 && (meta->max_args < 0 || meta->max_args >= 1)) {
                    ASTNode *wrapper = ast_new_op(meta->type);
                    if (wrapper) {
                        ASTNode *target_clone = evo_ast_clone(target);
                        if (target_clone && ast_add_child(wrapper, target_clone)) {
                            evo_replace_subtree(mutant, target, wrapper);
                        } else {
                            ast_free(wrapper);
                            ast_free(target_clone);
                        }
                    }
                    break;
                }
            }
        }
    } else {
        // Replace subtree with random
        int target_idx = random_range(1, node_count - 1);
        ASTNode *target = evo_random_subtree(mutant, &target_idx);

        if (target) {
            int new_depth = x->max_depth - evo_ast_depth(mutant) + 1;
            if (new_depth < 1) new_depth = 1;
            ASTNode *replacement = evolve_random_ast(x, new_depth);
            if (replacement) {
                if (!evo_replace_subtree(mutant, target, replacement)) {
                    ast_free(replacement);
                }
            }
        }
    }

    // Validate
    if (!evo_ast_valid(mutant)) {
        ast_free(mutant);
        return evo_ast_clone(node);
    }

    // Check depth
    if (evo_ast_depth(mutant) > x->max_depth) {
        ast_free(mutant);
        return evo_ast_clone(node);
    }

    return mutant;
}

// Combined mutation
static ASTNode* evolve_mutate(t_alien_evolve *x, ASTNode *node) {
    ASTNode *mutant = evo_ast_clone(node);
    if (!mutant) return NULL;

    // Point mutation
    if (random_range(0, 99) < 70) {
        evolve_mutate_point(x, mutant);
    }

    // Structural mutation
    if (random_range(0, 99) < 30) {
        ASTNode *new_mutant = evolve_mutate_structure(x, mutant);
        ast_free(mutant);
        mutant = new_mutant;
    }

    return mutant;
}

// ============================================================================
// FITNESS EVALUATION
// ============================================================================

static float evolve_min_corpus_distance(t_alien_evolve *x, ASTNode *node) {
    float min_dist = 1e9;
    for (int i = 0; i < x->corpus_count; i++) {
        float d = (float)evo_ast_distance(node, x->corpus[i].ast);
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

static float evolve_evaluate_fitness(t_alien_evolve *x, ASTNode *node) {
    if (!node) return 0;

    float score = 0;

    // Complexity metrics
    int depth = evo_ast_depth(node);
    int count = evo_ast_node_count(node);
    int ops = evo_ast_operator_count(node);

    // Reward moderate depth (peak at 3)
    score += (float)evo_gaussian(depth, 3.0, 1.5) * 0.25f;

    // Reward moderate size (peak at 8-12 nodes)
    score += (float)evo_gaussian(count, 10.0, 5.0) * 0.25f;

    // Reward having some operators (at least 1)
    if (ops >= 1) score += 0.1f;
    if (ops >= 2) score += 0.05f;

    // Novelty bonus: distance from corpus
    if (x->corpus_count > 0) {
        float min_dist = evolve_min_corpus_distance(x, node);
        float novelty = 1.0f - expf(-min_dist / 5.0f);
        score += novelty * (x->diversity_weight / 100.0f) * 0.3f;
    }

    // Parsimony: slight penalty for excessive size
    score += (1.0f / (1.0f + 0.02f * count)) * 0.1f;

    // Penalty for too deep
    if (depth > x->max_depth) {
        score -= 0.2f * (depth - x->max_depth);
    }

    return score;
}

// ============================================================================
// POPULATION MANAGEMENT
// ============================================================================

static void evolve_free_individual(Individual *ind) {
    if (ind->genome) {
        ast_free(ind->genome);
        ind->genome = NULL;
    }
    if (ind->source) {
        ALIEN_FREE(ind->source, strlen(ind->source) + 1);
        ind->source = NULL;
    }
}

static void evolve_init_population(t_alien_evolve *x) {
    // Free existing
    for (int i = 0; i < x->pop_size; i++) {
        evolve_free_individual(&x->population[i]);
    }

    // Initialize with random individuals
    for (int i = 0; i < x->pop_size; i++) {
        x->population[i].genome = evolve_random_ast(x, x->max_depth);
        x->population[i].source = evo_ast_serialize(x->population[i].genome);
        x->population[i].fitness = evolve_evaluate_fitness(x, x->population[i].genome);
    }

    x->generation = 0;
}

// Tournament selection
static Individual* evolve_tournament_select(t_alien_evolve *x) {
    Individual *best = NULL;

    for (int i = 0; i < x->tournament_size; i++) {
        int idx = random_range(0, x->pop_size - 1);
        if (!best || x->population[idx].fitness > best->fitness) {
            best = &x->population[idx];
        }
    }

    return best;
}

// Compare for sorting (descending fitness)
static int fitness_cmp(const void *a, const void *b) {
    const Individual *ia = (const Individual*)a;
    const Individual *ib = (const Individual*)b;
    if (ib->fitness > ia->fitness) return 1;
    if (ib->fitness < ia->fitness) return -1;
    return 0;
}

// Run one generation of evolution
static void evolve_step(t_alien_evolve *x) {
    // Sort by fitness
    qsort(x->population, x->pop_size, sizeof(Individual), fitness_cmp);

    // Create new generation
    Individual *new_pop = (Individual*)ALIEN_MALLOC(sizeof(Individual) * x->pop_capacity);
    if (!new_pop) return;
    memset(new_pop, 0, sizeof(Individual) * x->pop_capacity);

    int new_count = 0;

    // Elitism: keep top individuals
    for (int i = 0; i < x->elite_count && i < x->pop_size; i++) {
        new_pop[new_count].genome = evo_ast_clone(x->population[i].genome);
        new_pop[new_count].source = evo_ast_serialize(new_pop[new_count].genome);
        new_pop[new_count].fitness = x->population[i].fitness;
        new_count++;
    }

    // Generate rest through crossover/mutation
    while (new_count < x->pop_size) {
        ASTNode *child = NULL;

        if (random_range(0, 99) < (int)x->crossover_rate) {
            // Crossover
            Individual *p1 = evolve_tournament_select(x);
            Individual *p2 = evolve_tournament_select(x);
            child = evolve_crossover(x, p1->genome, p2->genome);
        } else {
            // Clone and mutate
            Individual *parent = evolve_tournament_select(x);
            child = evo_ast_clone(parent->genome);
        }

        if (random_range(0, 99) < (int)x->mutation_rate) {
            ASTNode *mutated = evolve_mutate(x, child);
            ast_free(child);
            child = mutated;
        }

        if (child) {
            new_pop[new_count].genome = child;
            new_pop[new_count].source = evo_ast_serialize(child);
            new_pop[new_count].fitness = evolve_evaluate_fitness(x, child);
            new_count++;
        }
    }

    // Replace old population
    for (int i = 0; i < x->pop_size; i++) {
        evolve_free_individual(&x->population[i]);
    }
    ALIEN_FREE(x->population, sizeof(Individual) * x->pop_capacity);
    x->population = new_pop;

    x->generation++;
    evolve_corpus_age(x);
}

// ============================================================================
// PATTERN GENERATION
// ============================================================================

static char* evolve_generate_pattern(t_alien_evolve *x) {
    char *result = NULL;

    switch (x->mode) {
        case MODE_EVOLVE: {
            // Run evolution step
            evolve_step(x);

            // Return best individual
            qsort(x->population, x->pop_size, sizeof(Individual), fitness_cmp);
            if (x->population[0].source) {
                size_t len = strlen(x->population[0].source) + 1;
                result = (char*)ALIEN_MALLOC(len);
                if (result) memcpy(result, x->population[0].source, len);
            }

            // Add to corpus if novel
            if (result && x->population[0].fitness > 0.5f) {
                evolve_corpus_add(x, result, 0);
            }
            break;
        }

        case MODE_RANDOM: {
            ASTNode *ast = evolve_random_ast(x, x->max_depth);
            if (ast) {
                result = evo_ast_serialize(ast);
                ast_free(ast);
            }
            break;
        }

        case MODE_NGRAM: {
            evolve_rebuild_ngram(x);
            if (x->corpus_count > 0) {
                // Start with random operator from corpus
                int start_idx = random_range(0, x->corpus_count - 1);
                const char *seed = x->corpus[start_idx].source;

                // Find first operator
                const char *paren = strchr(seed, '(');
                if (paren) {
                    char start[32];
                    int i = 0;
                    start[i++] = '(';
                    paren++;
                    while (*paren && !isspace(*paren) && *paren != ')' && i < 30) {
                        start[i++] = *paren++;
                    }
                    start[i++] = ' ';
                    start[i] = '\0';

                    result = evolve_ngram_complete(x, start, MAX_PATTERN_LEN);

                    // Validate
                    if (result) {
                        Token tokens[256];
                        int tc = tokenize(result, tokens, 256);
                        ASTNode *ast = (tc > 0) ? parse(tokens, tc) : NULL;
                        if (!ast || !evo_ast_valid(ast)) {
                            ALIEN_FREE(result, strlen(result) + 1);
                            result = NULL;
                        }
                        ast_free(ast);
                    }
                }
            }

            // Fallback to random
            if (!result) {
                ASTNode *ast = evolve_random_ast(x, x->max_depth);
                if (ast) {
                    result = evo_ast_serialize(ast);
                    ast_free(ast);
                }
            }
            break;
        }
    }

    return result;
}

// ============================================================================
// PD EXTERNAL INTERFACE
// ============================================================================

static void evolve_output_pattern(t_alien_evolve *x, const char *pattern) {
    if (!pattern) return;

    Token tokens[256];
    int token_count = tokenize(pattern, tokens, 256);
    if (token_count <= 0) return;

    int num_atoms = 0;
    for (int i = 0; i < token_count && tokens[i].type != TOK_EOF; i++) {
        num_atoms++;
    }
    if (num_atoms <= 0 || num_atoms > 200) return;

    t_atom atoms[200];
    int ai = 0;

    for (int i = 0; i < token_count && tokens[i].type != TOK_EOF && ai < 200; i++) {
        switch (tokens[i].type) {
            case TOK_LPAREN:
                SETSYMBOL(&atoms[ai++], gensym("("));
                break;
            case TOK_RPAREN:
                SETSYMBOL(&atoms[ai++], gensym(")"));
                break;
            case TOK_NUMBER:
                SETFLOAT(&atoms[ai++], (t_float)tokens[i].value.number);
                break;
            case TOK_HYPHEN:
                SETSYMBOL(&atoms[ai++], gensym("-"));
                break;
            case TOK_SYMBOL:
                SETSYMBOL(&atoms[ai++], gensym(tokens[i].value.symbol));
                break;
            default:
                break;
        }
    }

    if (ai > 0) {
        outlet_list(x->out_pattern, &s_list, ai, atoms);
    }
}

static void evolve_bang(t_alien_evolve *x) {
    // Generate random AST and serialize
    ASTNode *ast = evolve_random_ast(x, x->max_depth);
    if (!ast) return;

    char *pattern = evo_ast_serialize(ast);
    ast_free(ast);

    if (pattern) {
        evolve_output_pattern(x, pattern);
        freebytes(pattern, strlen(pattern) + 1);
    }
}

static void evolve_generate(t_alien_evolve *x, t_floatarg n) {
    int count = (int)n;
    if (count < 1) count = 1;
    if (count > 100) count = 100;

    for (int i = 0; i < count; i++) {
        evolve_bang(x);
    }
}

static void evolve_seed(t_alien_evolve *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;

    // Reconstruct pattern from atoms
    char pattern[MAX_PATTERN_LEN];
    int pos = 0;

    for (int i = 0; i < argc && pos < MAX_PATTERN_LEN - 32; i++) {
        if (i > 0) pattern[pos++] = ' ';

        if (argv[i].a_type == A_FLOAT) {
            pos += snprintf(pattern + pos, MAX_PATTERN_LEN - pos, "%d", (int)atom_getfloat(&argv[i]));
        } else if (argv[i].a_type == A_SYMBOL) {
            const char *sym = atom_getsymbol(&argv[i])->s_name;
            pos += snprintf(pattern + pos, MAX_PATTERN_LEN - pos, "%s", sym);
        }
    }
    pattern[pos] = '\0';

    if (evolve_corpus_add(x, pattern, 1)) {
        t_atom info[2];
        SETSYMBOL(&info[0], gensym("seeded"));
        SETSYMBOL(&info[1], gensym(pattern));
        outlet_list(x->out_info, &s_list, 2, info);
    } else {
        outlet_symbol(x->out_info, gensym("seed_failed"));
    }
}

static void evolve_load(t_alien_evolve *x, t_symbol *filename) {
    FILE *f = fopen(filename->s_name, "r");
    if (!f) {
        pd_error(x, "alien_evolve: can't open %s", filename->s_name);
        return;
    }

    char line[MAX_PATTERN_LEN];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        // Strip newline
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';

        // Skip empty lines and comments
        if (line[0] == '\0' || line[0] == '#') continue;

        if (evolve_corpus_add(x, line, 1)) {
            count++;
        }
    }

    fclose(f);

    t_atom info[2];
    SETSYMBOL(&info[0], gensym("loaded"));
    SETFLOAT(&info[1], count);
    outlet_list(x->out_info, &s_list, 2, info);
}

static void evolve_save(t_alien_evolve *x, t_symbol *filename) {
    FILE *f = fopen(filename->s_name, "w");
    if (!f) {
        pd_error(x, "alien_evolve: can't write %s", filename->s_name);
        return;
    }

    int count = 0;
    for (int i = 0; i < x->corpus_count; i++) {
        fprintf(f, "%s\n", x->corpus[i].source);
        count++;
    }

    fclose(f);

    t_atom info[2];
    SETSYMBOL(&info[0], gensym("saved"));
    SETFLOAT(&info[1], count);
    outlet_list(x->out_info, &s_list, 2, info);
}

static void evolve_clear(t_alien_evolve *x) {
    // Clear non-protected entries
    int i = 0;
    while (i < x->corpus_count) {
        if (!x->corpus[i].protected) {
            evolve_free_corpus_entry(&x->corpus[i]);
            for (int j = i; j < x->corpus_count - 1; j++) {
                x->corpus[j] = x->corpus[j + 1];
            }
            x->corpus_count--;
        } else {
            i++;
        }
    }
    x->ngram_built = 0;

    outlet_symbol(x->out_info, gensym("cleared"));
}

// Configuration methods
static void evolve_popsize(t_alien_evolve *x, t_floatarg n) {
    int new_size = (int)n;
    if (new_size < 10) new_size = 10;
    if (new_size > 1000) new_size = 1000;

    if (new_size > x->pop_capacity) {
        Individual *new_pop = (Individual*)ALIEN_REALLOC(x->population,
            sizeof(Individual) * x->pop_capacity,
            sizeof(Individual) * new_size);
        if (!new_pop) return;
        x->population = new_pop;
        x->pop_capacity = new_size;
    }

    x->pop_size = new_size;
    evolve_init_population(x);
}

static void evolve_maxdepth(t_alien_evolve *x, t_floatarg n) {
    x->max_depth = (int)n;
    if (x->max_depth < 1) x->max_depth = 1;
    if (x->max_depth > 10) x->max_depth = 10;
}

static void evolve_maxsize(t_alien_evolve *x, t_floatarg n) {
    x->corpus_max = (int)n;
    if (x->corpus_max < 10) x->corpus_max = 10;
    if (x->corpus_max > 10000) x->corpus_max = 10000;
}

static void evolve_mutation(t_alien_evolve *x, t_floatarg n) {
    x->mutation_rate = n;
    if (x->mutation_rate < 0) x->mutation_rate = 0;
    if (x->mutation_rate > 100) x->mutation_rate = 100;
}

static void evolve_set_crossover(t_alien_evolve *x, t_floatarg n) {
    x->crossover_rate = n;
    if (x->crossover_rate < 0) x->crossover_rate = 0;
    if (x->crossover_rate > 100) x->crossover_rate = 100;
}

static void evolve_elite(t_alien_evolve *x, t_floatarg n) {
    x->elite_count = (int)n;
    if (x->elite_count < 0) x->elite_count = 0;
    if (x->elite_count > x->pop_size / 2) x->elite_count = x->pop_size / 2;
}

static void evolve_tournament(t_alien_evolve *x, t_floatarg n) {
    x->tournament_size = (int)n;
    if (x->tournament_size < 2) x->tournament_size = 2;
    if (x->tournament_size > x->pop_size) x->tournament_size = x->pop_size;
}

static void evolve_diversity(t_alien_evolve *x, t_floatarg n) {
    x->diversity_weight = n;
    if (x->diversity_weight < 0) x->diversity_weight = 0;
    if (x->diversity_weight > 100) x->diversity_weight = 100;
}

static void evolve_mode(t_alien_evolve *x, t_symbol *mode) {
    if (strcmp(mode->s_name, "evolve") == 0) {
        x->mode = MODE_EVOLVE;
    } else if (strcmp(mode->s_name, "random") == 0) {
        x->mode = MODE_RANDOM;
    } else if (strcmp(mode->s_name, "ngram") == 0) {
        x->mode = MODE_NGRAM;
    } else {
        pd_error(x, "alien_evolve: unknown mode %s (use evolve/random/ngram)", mode->s_name);
    }
}

// Info methods
static void evolve_stats(t_alien_evolve *x) {
    t_atom info[8];
    SETSYMBOL(&info[0], gensym("corpus"));
    SETFLOAT(&info[1], x->corpus_count);
    SETSYMBOL(&info[2], gensym("gen"));
    SETFLOAT(&info[3], x->generation);
    SETSYMBOL(&info[4], gensym("pop"));
    SETFLOAT(&info[5], x->pop_size);
    SETSYMBOL(&info[6], gensym("mode"));

    const char *mode_name = "evolve";
    if (x->mode == MODE_RANDOM) mode_name = "random";
    if (x->mode == MODE_NGRAM) mode_name = "ngram";
    SETSYMBOL(&info[7], gensym(mode_name));

    outlet_list(x->out_info, &s_list, 8, info);
}

static void evolve_best(t_alien_evolve *x, t_floatarg n) {
    int count = (int)n;
    if (count < 1) count = 1;
    if (count > x->pop_size) count = x->pop_size;

    qsort(x->population, x->pop_size, sizeof(Individual), fitness_cmp);

    for (int i = 0; i < count; i++) {
        if (x->population[i].source) {
            t_atom info[3];
            SETFLOAT(&info[0], i + 1);
            SETFLOAT(&info[1], x->population[i].fitness);
            SETSYMBOL(&info[2], gensym(x->population[i].source));
            outlet_list(x->out_info, &s_list, 3, info);
        }
    }
}

static void evolve_corpus(t_alien_evolve *x) {
    for (int i = 0; i < x->corpus_count; i++) {
        t_atom info[4];
        SETFLOAT(&info[0], i);
        SETFLOAT(&info[1], x->corpus[i].fitness);
        SETFLOAT(&info[2], x->corpus[i].protected);
        SETSYMBOL(&info[3], gensym(x->corpus[i].source));
        outlet_list(x->out_info, &s_list, 4, info);
    }
}

// ============================================================================
// SETUP / CLEANUP
// ============================================================================

static void *evolve_new(t_floatarg pop, t_floatarg depth) {
    t_alien_evolve *x = (t_alien_evolve *)pd_new(alien_evolve_class);

    // Initialize configuration
    x->pop_size = (pop > 0) ? (int)pop : DEFAULT_POP_SIZE;
    x->max_depth = (depth > 0) ? (int)depth : DEFAULT_MAX_DEPTH;
    x->corpus_max = DEFAULT_CORPUS_MAX;
    x->mutation_rate = DEFAULT_MUTATION;
    x->crossover_rate = DEFAULT_CROSSOVER;
    x->elite_count = DEFAULT_ELITE;
    x->tournament_size = DEFAULT_TOURNAMENT;
    x->diversity_weight = DEFAULT_DIVERSITY;
    x->mode = MODE_EVOLVE;
    x->generation = 0;

    // Allocate corpus
    x->corpus_capacity = 64;
    x->corpus_count = 0;
    x->corpus = (CorpusEntry*)ALIEN_MALLOC(sizeof(CorpusEntry) * x->corpus_capacity);
    memset(x->corpus, 0, sizeof(CorpusEntry) * x->corpus_capacity);

    // Initialize n-gram
    x->ngram_text = NULL;
    x->ngram_len = 0;
    x->suffix_array = NULL;
    x->ngram_built = 0;

    // Allocate population
    x->pop_capacity = x->pop_size;
    x->population = (Individual*)ALIEN_MALLOC(sizeof(Individual) * x->pop_capacity);
    memset(x->population, 0, sizeof(Individual) * x->pop_capacity);

    // Initialize population
    evolve_init_population(x);

    // Create outlets
    x->out_pattern = outlet_new(&x->x_obj, &s_symbol);
    x->out_info = outlet_new(&x->x_obj, &s_list);

    return x;
}

static void evolve_free(t_alien_evolve *x) {
    // Free corpus
    for (int i = 0; i < x->corpus_count; i++) {
        evolve_free_corpus_entry(&x->corpus[i]);
    }
    ALIEN_FREE(x->corpus, sizeof(CorpusEntry) * x->corpus_capacity);

    // Free n-gram
    if (x->ngram_text) {
        ALIEN_FREE(x->ngram_text, x->ngram_len + 1);
    }
    if (x->suffix_array) {
        ALIEN_FREE(x->suffix_array, sizeof(uint32_t) * x->ngram_len);
    }

    // Free population
    for (int i = 0; i < x->pop_size; i++) {
        evolve_free_individual(&x->population[i]);
    }
    ALIEN_FREE(x->population, sizeof(Individual) * x->pop_capacity);

    // Outlets freed automatically
}

void alien_evolve_setup(void) {
    alien_evolve_class = class_new(gensym("alien_evolve"),
        (t_newmethod)evolve_new,
        (t_method)evolve_free,
        sizeof(t_alien_evolve),
        CLASS_DEFAULT,
        A_DEFFLOAT, A_DEFFLOAT, 0);

    // Core messages
    class_addbang(alien_evolve_class, evolve_bang);
    class_addmethod(alien_evolve_class, (t_method)evolve_generate,
        gensym("generate"), A_DEFFLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_seed,
        gensym("seed"), A_GIMME, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_load,
        gensym("load"), A_SYMBOL, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_save,
        gensym("save"), A_SYMBOL, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_clear,
        gensym("clear"), 0);

    // Configuration
    class_addmethod(alien_evolve_class, (t_method)evolve_popsize,
        gensym("popsize"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_maxdepth,
        gensym("maxdepth"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_maxsize,
        gensym("maxsize"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_mutation,
        gensym("mutation"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_set_crossover,
        gensym("crossover"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_elite,
        gensym("elite"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_tournament,
        gensym("tournament"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_diversity,
        gensym("diversity"), A_FLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_mode,
        gensym("mode"), A_SYMBOL, 0);

    // Info
    class_addmethod(alien_evolve_class, (t_method)evolve_stats,
        gensym("stats"), 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_best,
        gensym("best"), A_DEFFLOAT, 0);
    class_addmethod(alien_evolve_class, (t_method)evolve_corpus,
        gensym("corpus"), 0);

    post("alien_evolve: pattern evolution external v0.1");
}
