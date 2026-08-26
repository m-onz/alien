/*
 * ns_grid.h - 2D grid statistics → 6-dim BC
 *
 * One projector for one specific domain (binary 2D grids). Lives outside
 * ns_core.h so that core stays purely cross-domain primitives and other
 * projectors don't pull grid code into their translation unit.
 *
 * Treats input as binary (>=0.5 → 1, <0.5 → 0).
 *
 * Output (always 6 floats, in this order):
 *   0  density          mean of cells, in [0, 1]
 *   1  spatial_entropy  Shannon entropy over 2x2 sub-block density bins,
 *                       normalised to [0, 1]
 *   2  sym_h            horizontal symmetry: 1 - mean|top - flipped(bottom)|
 *   3  sym_v            vertical symmetry:   1 - mean|left - flipped(right)|
 *   4  components       4-connected component count, normalised by W*H/2
 *   5  mean_comp_size   mean component size / (W*H), in [0, 1]
 *
 * Header-only, like ns_core.h. Depends only on math.h and string.h.
 */

#ifndef NS_GRID_H
#define NS_GRID_H

#include <math.h>
#include <string.h>

#define NS_GRID_MAX 64

static inline void ns_grid_stats(const float *grid, int w, int h, float *out6) {
    if (!grid || !out6 || w <= 0 || h <= 0 || w > NS_GRID_MAX || h > NS_GRID_MAX) {
        for (int i = 0; i < 6; i++) out6[i] = 0.0f;
        return;
    }
    int N = w * h;

    /* 1. Density. */
    int alive = 0;
    for (int i = 0; i < N; i++) if (grid[i] >= 0.5f) alive++;
    float density = (float)alive / (float)N;

    /* 2. Spatial entropy: divide grid into 2x2 blocks (or 1x1 if too small)
     *    and compute Shannon entropy over the distribution of block densities
     *    bucketed into 5 bins {0, 1/4, 2/4, 3/4, 1}. Normalised by log2(5). */
    int bins[5] = {0, 0, 0, 0, 0};
    int total_blocks = 0;
    for (int by = 0; by + 1 < h; by += 2) {
        for (int bx = 0; bx + 1 < w; bx += 2) {
            int s = 0;
            s += (grid[(by    ) * w + (bx    )] >= 0.5f);
            s += (grid[(by    ) * w + (bx + 1)] >= 0.5f);
            s += (grid[(by + 1) * w + (bx    )] >= 0.5f);
            s += (grid[(by + 1) * w + (bx + 1)] >= 0.5f);
            bins[s]++;  /* s in [0, 4] → bin index */
            total_blocks++;
        }
    }
    float entropy = 0.0f;
    if (total_blocks > 0) {
        for (int i = 0; i < 5; i++) {
            if (bins[i] > 0) {
                float p = (float)bins[i] / (float)total_blocks;
                entropy -= p * (logf(p) / logf(2.0f));
            }
        }
        entropy /= (logf(5.0f) / logf(2.0f));
    }

    /* 3. Horizontal symmetry: top half vs vertically-flipped bottom half. */
    float sym_h = 1.0f;
    {
        int half = h / 2;
        if (half > 0) {
            float diff = 0.0f;
            int n = 0;
            for (int y = 0; y < half; y++) {
                for (int x = 0; x < w; x++) {
                    float a = (grid[y * w + x] >= 0.5f) ? 1.0f : 0.0f;
                    float b = (grid[(h - 1 - y) * w + x] >= 0.5f) ? 1.0f : 0.0f;
                    diff += fabsf(a - b);
                    n++;
                }
            }
            sym_h = 1.0f - (n > 0 ? diff / (float)n : 0.0f);
        }
    }

    /* 4. Vertical symmetry: left half vs horizontally-flipped right half. */
    float sym_v = 1.0f;
    {
        int half = w / 2;
        if (half > 0) {
            float diff = 0.0f;
            int n = 0;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < half; x++) {
                    float a = (grid[y * w + x] >= 0.5f) ? 1.0f : 0.0f;
                    float b = (grid[y * w + (w - 1 - x)] >= 0.5f) ? 1.0f : 0.0f;
                    diff += fabsf(a - b);
                    n++;
                }
            }
            sym_v = 1.0f - (n > 0 ? diff / (float)n : 0.0f);
        }
    }

    /* 5–6. Connected components (4-connected) over the live cells.
     *      Iterative flood fill on a stack of indices. */
    int component_count = 0;
    int total_size = 0;
    {
        char visited[NS_GRID_MAX * NS_GRID_MAX];
        memset(visited, 0, sizeof(visited));
        int stack[NS_GRID_MAX * NS_GRID_MAX];
        for (int seed = 0; seed < N; seed++) {
            if (visited[seed]) continue;
            if (grid[seed] < 0.5f) { visited[seed] = 1; continue; }
            int sp = 0;
            stack[sp++] = seed;
            visited[seed] = 1;
            int size = 0;
            while (sp > 0) {
                int idx = stack[--sp];
                size++;
                int x = idx % w;
                int y = idx / w;
                int neighbors[4] = {
                    (x > 0)     ? idx - 1 : -1,
                    (x < w - 1) ? idx + 1 : -1,
                    (y > 0)     ? idx - w : -1,
                    (y < h - 1) ? idx + w : -1,
                };
                for (int ni = 0; ni < 4; ni++) {
                    int n = neighbors[ni];
                    if (n < 0 || visited[n]) continue;
                    visited[n] = 1;
                    if (grid[n] >= 0.5f) stack[sp++] = n;
                }
            }
            component_count++;
            total_size += size;
        }
    }
    float comp_norm = (float)component_count / ((float)N * 0.5f);
    if (comp_norm > 1.0f) comp_norm = 1.0f;
    float mean_comp_size = (component_count > 0)
        ? ((float)total_size / (float)component_count) / (float)N
        : 0.0f;

    out6[0] = density;
    out6[1] = entropy;
    out6[2] = sym_h;
    out6[3] = sym_v;
    out6[4] = comp_norm;
    out6[5] = mean_comp_size;
}

#endif /* NS_GRID_H */
