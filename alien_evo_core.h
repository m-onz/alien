/*
 * alien_evo_core.h - Evolution infrastructure for alien pattern language
 *
 * Provides operator arity table, AST utilities for cloning, serialization,
 * hashing, depth/size measurement, and tree manipulation for genetic operations.
 */

#ifndef ALIEN_EVO_CORE_H
#define ALIEN_EVO_CORE_H

#include "alien_core.h"
#include <stdint.h>

// ============================================================================
// OPERATOR METADATA
// ============================================================================

typedef struct {
    NodeType type;
    const char *name;
    int min_args;
    int max_args;  // -1 for variadic
    int is_leaf_producer;  // Does this typically produce leaf values?
} OpMeta;

// Operator metadata table
static const OpMeta g_op_meta[] = {
    { NODE_SEQ,        "seq",        1, -1, 0 },
    { NODE_REP,        "rep",        2, -1, 0 },
    { NODE_ADD,        "add",        2,  2, 0 },
    { NODE_MUL,        "mul",        2,  2, 0 },
    { NODE_MOD,        "mod",        2,  2, 0 },
    { NODE_SCALE,      "scale",      5,  5, 0 },
    { NODE_CLAMP,      "clamp",      3,  3, 0 },
    { NODE_EUCLID,     "euclid",     2,  3, 1 },
    { NODE_BJORK,      "bjork",      2,  2, 1 },
    { NODE_SUBDIV,     "subdiv",     2,  2, 0 },
    { NODE_REVERSE,    "reverse",    1,  1, 0 },
    { NODE_ROTATE,     "rotate",     2,  2, 0 },
    { NODE_PALINDROME, "palindrome", 1,  1, 0 },
    { NODE_MIRROR,     "mirror",     1,  1, 0 },
    { NODE_INTERLEAVE, "interleave", 2,  2, 0 },
    { NODE_SHUFFLE,    "shuffle",    1,  1, 0 },
    { NODE_TAKE,       "take",       2,  2, 0 },
    { NODE_DROP,       "drop",       2,  2, 0 },
    { NODE_EVERY,      "every",      2,  2, 0 },
    { NODE_SLICE,      "slice",      3,  3, 0 },
    { NODE_FILTER,     "filter",     1,  1, 0 },
    { NODE_CHOOSE,     "choose",     1, -1, 0 },
    { NODE_RAND,       "rand",       3,  3, 1 },
    { NODE_PROB,       "prob",       2,  2, 0 },
    { NODE_MAYBE,      "maybe",      3,  3, 0 },
    { NODE_RANGE,      "range",      2,  3, 1 },
    { NODE_RAMP,       "ramp",       3,  3, 1 },
    { NODE_DRUNK,      "drunk",      3,  3, 1 },
    { NODE_CYCLE,      "cycle",      2,  2, 0 },
    { NODE_GROW,       "grow",       1,  1, 0 },
    { NODE_DEGRADE,    "degrade",    2,  2, 0 },
    { NODE_TRANSPOSE,  "transpose",  2,  2, 0 },
    { NODE_QUANTIZE,   "quantize",   2,  2, 0 },
    { NODE_CHORD,      "chord",      2,  2, 1 },
    { NODE_ARP,        "arp",        3,  3, 0 },
    { NODE_DELAY,      "delay",      2,  2, 0 },
    { NODE_GATE,       "gate",       2,  2, 0 },
};

#define OP_META_COUNT (sizeof(g_op_meta) / sizeof(g_op_meta[0]))

static const OpMeta* evo_get_op_meta(NodeType type) {
    for (size_t i = 0; i < OP_META_COUNT; i++) {
        if (g_op_meta[i].type == type) {
            return &g_op_meta[i];
        }
    }
    return NULL;
}

static const OpMeta* evo_get_op_meta_by_name(const char *name) {
    for (size_t i = 0; i < OP_META_COUNT; i++) {
        if (strcmp(g_op_meta[i].name, name) == 0) {
            return &g_op_meta[i];
        }
    }
    return NULL;
}

static const char* evo_get_op_name(NodeType type) {
    const OpMeta *meta = evo_get_op_meta(type);
    return meta ? meta->name : NULL;
}

// ============================================================================
// AST CLONING
// ============================================================================

static ASTNode* evo_ast_clone(ASTNode *node) {
    if (!node) return NULL;

    if (node->type == NODE_NUMBER) {
        return ast_new_number(node->data.number);
    }

    if (node->type == NODE_HYPHEN) {
        return ast_new_hyphen();
    }

    // Operator node
    ASTNode *clone = ast_new_op(node->type);
    if (!clone) return NULL;

    for (int i = 0; i < node->data.op.child_count; i++) {
        ASTNode *child_clone = evo_ast_clone(node->data.op.children[i]);
        if (!child_clone) {
            ast_free(clone);
            return NULL;
        }
        if (!ast_add_child(clone, child_clone)) {
            ast_free(child_clone);
            ast_free(clone);
            return NULL;
        }
    }

    return clone;
}

// ============================================================================
// AST SERIALIZATION
// ============================================================================

// Calculate buffer size needed for serialization
static int evo_ast_serial_len(ASTNode *node) {
    if (!node) return 0;

    if (node->type == NODE_NUMBER) {
        char buf[32];
        return snprintf(buf, sizeof(buf), "%d", node->data.number);
    }

    if (node->type == NODE_HYPHEN) {
        return 1;  // "-"
    }

    // Operator: (name child1 child2 ...)
    const char *name = evo_get_op_name(node->type);
    if (!name) return 0;

    int len = 1 + (int)strlen(name);  // "(" + name
    for (int i = 0; i < node->data.op.child_count; i++) {
        len += 1 + evo_ast_serial_len(node->data.op.children[i]);  // " " + child
    }
    len += 1;  // ")"
    return len;
}

static int evo_ast_serialize_impl(ASTNode *node, char *buf, int bufsize, int pos) {
    if (!node || pos >= bufsize - 1) return pos;

    if (node->type == NODE_NUMBER) {
        int written = snprintf(buf + pos, bufsize - pos, "%d", node->data.number);
        return pos + written;
    }

    if (node->type == NODE_HYPHEN) {
        if (pos < bufsize - 1) buf[pos++] = '-';
        return pos;
    }

    const char *name = evo_get_op_name(node->type);
    if (!name) return pos;

    if (pos < bufsize - 1) buf[pos++] = '(';
    int name_len = (int)strlen(name);
    if (pos + name_len < bufsize) {
        memcpy(buf + pos, name, name_len);
        pos += name_len;
    }

    for (int i = 0; i < node->data.op.child_count; i++) {
        if (pos < bufsize - 1) buf[pos++] = ' ';
        pos = evo_ast_serialize_impl(node->data.op.children[i], buf, bufsize, pos);
    }

    if (pos < bufsize - 1) buf[pos++] = ')';
    return pos;
}

// Serialize AST to string. Returns allocated string (caller must free).
static char* evo_ast_serialize(ASTNode *node) {
    if (!node) return NULL;

    int len = evo_ast_serial_len(node) + 1;
    char *buf = (char*)ALIEN_MALLOC(len);
    if (!buf) return NULL;

    int final_pos = evo_ast_serialize_impl(node, buf, len, 0);
    buf[final_pos] = '\0';
    return buf;
}

// ============================================================================
// AST HASHING (FNV-1a)
// ============================================================================

#define FNV_OFFSET_BASIS 2166136261u
#define FNV_PRIME 16777619u

static uint32_t evo_fnv1a_update(uint32_t hash, const char *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint8_t)data[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint32_t evo_ast_hash_impl(ASTNode *node, uint32_t hash) {
    if (!node) return hash;

    if (node->type == NODE_NUMBER) {
        hash ^= (uint8_t)'N';
        hash *= FNV_PRIME;
        return evo_fnv1a_update(hash, (const char*)&node->data.number, sizeof(int));
    }

    if (node->type == NODE_HYPHEN) {
        hash ^= (uint8_t)'-';
        hash *= FNV_PRIME;
        return hash;
    }

    // Operator
    hash ^= (uint8_t)'O';
    hash *= FNV_PRIME;
    hash ^= (uint8_t)node->type;
    hash *= FNV_PRIME;

    for (int i = 0; i < node->data.op.child_count; i++) {
        hash = evo_ast_hash_impl(node->data.op.children[i], hash);
    }

    return hash;
}

static uint32_t evo_ast_hash(ASTNode *node) {
    return evo_ast_hash_impl(node, FNV_OFFSET_BASIS);
}

// Quick string hash
static uint32_t evo_str_hash(const char *str) {
    if (!str) return 0;
    return evo_fnv1a_update(FNV_OFFSET_BASIS, str, strlen(str));
}

// ============================================================================
// AST METRICS
// ============================================================================

static int evo_ast_depth(ASTNode *node) {
    if (!node) return 0;
    if (node->type == NODE_NUMBER || node->type == NODE_HYPHEN) {
        return 1;
    }

    int max_child_depth = 0;
    for (int i = 0; i < node->data.op.child_count; i++) {
        int d = evo_ast_depth(node->data.op.children[i]);
        if (d > max_child_depth) max_child_depth = d;
    }
    return 1 + max_child_depth;
}

static int evo_ast_node_count(ASTNode *node) {
    if (!node) return 0;
    if (node->type == NODE_NUMBER || node->type == NODE_HYPHEN) {
        return 1;
    }

    int count = 1;  // This node
    for (int i = 0; i < node->data.op.child_count; i++) {
        count += evo_ast_node_count(node->data.op.children[i]);
    }
    return count;
}

static int evo_ast_operator_count(ASTNode *node) {
    if (!node) return 0;
    if (node->type == NODE_NUMBER || node->type == NODE_HYPHEN) {
        return 0;
    }

    int count = 1;  // This operator
    for (int i = 0; i < node->data.op.child_count; i++) {
        count += evo_ast_operator_count(node->data.op.children[i]);
    }
    return count;
}

// ============================================================================
// AST TREE UTILITIES
// ============================================================================

// Get a random subtree from an AST
static ASTNode* evo_random_subtree(ASTNode *node, int *index) {
    if (!node) return NULL;

    if (*index == 0) {
        return node;
    }
    (*index)--;

    if (node->type == NODE_NUMBER || node->type == NODE_HYPHEN) {
        return NULL;
    }

    for (int i = 0; i < node->data.op.child_count; i++) {
        ASTNode *result = evo_random_subtree(node->data.op.children[i], index);
        if (result) return result;
    }
    return NULL;
}

// Replace a subtree in a cloned tree (modifies in place)
// Returns 1 if replacement happened, 0 otherwise
static int evo_replace_subtree(ASTNode *root, ASTNode *target, ASTNode *replacement) {
    if (!root || !target || !replacement) return 0;
    if (root->type == NODE_NUMBER || root->type == NODE_HYPHEN) return 0;

    for (int i = 0; i < root->data.op.child_count; i++) {
        if (root->data.op.children[i] == target) {
            ast_free(root->data.op.children[i]);
            root->data.op.children[i] = replacement;
            return 1;
        }
        if (evo_replace_subtree(root->data.op.children[i], target, replacement)) {
            return 1;
        }
    }
    return 0;
}

// Find all number leaves in a tree
typedef struct {
    ASTNode **nodes;
    int count;
    int capacity;
} NodeList;

static NodeList* nodelist_new(void) {
    NodeList *list = (NodeList*)ALIEN_MALLOC(sizeof(NodeList));
    if (!list) return NULL;
    list->capacity = 16;
    list->count = 0;
    list->nodes = (ASTNode**)ALIEN_MALLOC(sizeof(ASTNode*) * list->capacity);
    if (!list->nodes) {
        ALIEN_FREE(list, sizeof(NodeList));
        return NULL;
    }
    return list;
}

static void nodelist_free(NodeList *list) {
    if (list) {
        if (list->nodes) ALIEN_FREE(list->nodes, sizeof(ASTNode*) * list->capacity);
        ALIEN_FREE(list, sizeof(NodeList));
    }
}

static int nodelist_add(NodeList *list, ASTNode *node) {
    if (list->count >= list->capacity) {
        int old_cap = list->capacity;
        int new_cap = list->capacity * 2;
        ASTNode **new_nodes = (ASTNode**)ALIEN_REALLOC(list->nodes,
            sizeof(ASTNode*) * old_cap, sizeof(ASTNode*) * new_cap);
        if (!new_nodes) return 0;
        list->nodes = new_nodes;
        list->capacity = new_cap;
    }
    list->nodes[list->count++] = node;
    return 1;
}

static void evo_collect_number_leaves(ASTNode *node, NodeList *list) {
    if (!node || !list) return;

    if (node->type == NODE_NUMBER) {
        nodelist_add(list, node);
        return;
    }

    if (node->type == NODE_HYPHEN) return;

    for (int i = 0; i < node->data.op.child_count; i++) {
        evo_collect_number_leaves(node->data.op.children[i], list);
    }
}

// ============================================================================
// AST COMPARISON / DISTANCE
// ============================================================================

// Simple tree edit distance approximation (not full Levenshtein)
static int evo_ast_distance(ASTNode *a, ASTNode *b) {
    if (!a && !b) return 0;
    if (!a || !b) return 10;  // One is null

    // Both are numbers
    if (a->type == NODE_NUMBER && b->type == NODE_NUMBER) {
        return abs(a->data.number - b->data.number) > 10 ? 1 : 0;
    }

    // One number, one not
    if (a->type == NODE_NUMBER || b->type == NODE_NUMBER) {
        return 5;
    }

    // Both hyphens
    if (a->type == NODE_HYPHEN && b->type == NODE_HYPHEN) {
        return 0;
    }

    // One hyphen, one not
    if (a->type == NODE_HYPHEN || b->type == NODE_HYPHEN) {
        return 3;
    }

    // Both operators
    int dist = (a->type != b->type) ? 3 : 0;

    int max_children = a->data.op.child_count > b->data.op.child_count
                       ? a->data.op.child_count : b->data.op.child_count;
    int min_children = a->data.op.child_count < b->data.op.child_count
                       ? a->data.op.child_count : b->data.op.child_count;

    // Extra children cost
    dist += (max_children - min_children) * 2;

    // Compare common children
    for (int i = 0; i < min_children; i++) {
        dist += evo_ast_distance(a->data.op.children[i], b->data.op.children[i]);
    }

    return dist;
}

// ============================================================================
// VALIDATION
// ============================================================================

// Validate that an AST follows arity constraints
static int evo_ast_valid(ASTNode *node) {
    if (!node) return 0;

    if (node->type == NODE_NUMBER || node->type == NODE_HYPHEN) {
        return 1;
    }

    const OpMeta *meta = evo_get_op_meta(node->type);
    if (!meta) return 0;

    int argc = node->data.op.child_count;
    if (argc < meta->min_args) return 0;
    if (meta->max_args >= 0 && argc > meta->max_args) return 0;

    for (int i = 0; i < argc; i++) {
        if (!evo_ast_valid(node->data.op.children[i])) return 0;
    }

    return 1;
}

// ============================================================================
// GAUSSIAN HELPER FOR FITNESS
// ============================================================================

static double evo_gaussian(double x, double mean, double sigma) {
    double diff = x - mean;
    return exp(-(diff * diff) / (2.0 * sigma * sigma));
}

#endif // ALIEN_EVO_CORE_H
