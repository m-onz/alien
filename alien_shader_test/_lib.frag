// _lib.frag - GLSL Helper Library for Alien AV Toolkit
// Include this in your shaders for easy primitives, transforms, colors, etc.

// ============================================================================
// CONSTANTS
// ============================================================================
#define PI 3.14159265359
#define TAU 6.28318530718
#define E 2.71828182846

// ============================================================================
// UTILITY
// ============================================================================
float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, 0.0, 1.0); }
vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }

float remap(float v, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (outMax - outMin) * (v - inMin) / (inMax - inMin);
}

float linearstep(float a, float b, float x) {
    return saturate((x - a) / (b - a));
}

// Map 0-1 param to custom range (useful for control params)
float param(float p, float lo, float hi) {
    return mix(lo, hi, p);
}

// ============================================================================
// RANDOM & NOISE
// ============================================================================
float hash(float p) {
    p = fract(p * 0.011);
    p *= p + 7.5;
    p *= p + p;
    return fract(p);
}

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

float hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

vec2 hash2(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}

vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}

// Value noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n = i.x + i.y * 57.0 + 113.0 * i.z;
    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                   mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                   mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}

// Fractal Brownian Motion
float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 5; ++i) {
        v += a * noise(p);
        p = rot * p * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float fbm3(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; ++i) {
        v += a * noise3(p);
        p = p * 2.0 + 0.5;
        a *= 0.5;
    }
    return v;
}

// Voronoi
vec2 voronoi(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);

    float minDist = 8.0;
    vec2 minPoint;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < minDist) {
                minDist = d;
                minPoint = n + g + o;
            }
        }
    }
    return vec2(sqrt(minDist), hash(minPoint));
}

// ============================================================================
// 2D TRANSFORMS
// ============================================================================
vec2 rot2(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

vec2 scale2(vec2 p, float s) {
    return p / s;
}

vec2 scale2(vec2 p, vec2 s) {
    return p / s;
}

// Polar coordinates
vec2 toPolar(vec2 p) {
    return vec2(length(p), atan(p.y, p.x));
}

vec2 fromPolar(vec2 p) {
    return vec2(p.x * cos(p.y), p.x * sin(p.y));
}

// Repeat space in 2D
vec2 rep2(vec2 p, vec2 c) {
    return mod(p + 0.5 * c, c) - 0.5 * c;
}

// Polar repeat (for radial symmetry)
vec2 repPolar(vec2 p, float n) {
    float angle = TAU / n;
    float a = atan(p.y, p.x) + angle * 0.5;
    a = mod(a, angle) - angle * 0.5;
    return vec2(cos(a), sin(a)) * length(p);
}

// ============================================================================
// 3D TRANSFORMS
// ============================================================================
vec3 rotX(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

vec3 rotZ(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}

// Rotate around arbitrary axis
vec3 rotAxis(vec3 p, vec3 axis, float a) {
    axis = normalize(axis);
    float c = cos(a), s = sin(a);
    return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1.0 - c);
}

// Repeat space
vec3 rep(vec3 p, vec3 c) {
    return mod(p + 0.5 * c, c) - 0.5 * c;
}

// Limited repeat
vec3 repLim(vec3 p, vec3 c, vec3 l) {
    return p - c * clamp(floor(p / c + 0.5), -l, l);
}

// ============================================================================
// 2D SDF PRIMITIVES
// ============================================================================
float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

float sdRect(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + r;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdHex(vec2 p, float r) {
    const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

float sdTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

float sdStar(vec2 p, float r, int n, float m) {
    float an = PI / float(n);
    float en = PI / m;
    vec2 acs = vec2(cos(an), sin(an));
    vec2 ecs = vec2(cos(en), sin(en));
    float bn = mod(atan(p.x, p.y), 2.0 * an) - an;
    p = length(p) * vec2(cos(bn), abs(sin(bn)));
    p -= r * acs;
    p += ecs * clamp(-dot(p, ecs), 0.0, r * acs.y / ecs.y);
    return length(p) * sign(p.x);
}

// ============================================================================
// 3D SDF PRIMITIVES
// ============================================================================
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - r;
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdCone(vec3 p, vec2 c, float h) {
    vec2 q = h * vec2(c.x / c.y, -1.0);
    vec2 w = vec2(length(p.xz), p.y);
    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

float sdPlane(vec3 p, vec3 n, float h) {
    return dot(p, n) + h;
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;
    vec3 q;
    if (3.0 * p.x < m) q = p.xyz;
    else if (3.0 * p.y < m) q = p.yzx;
    else if (3.0 * p.z < m) q = p.zxy;
    else return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3(q.x, q.y - s + k, q.z - k));
}

// ============================================================================
// SDF OPERATIONS
// ============================================================================
float opU(float d1, float d2) { return min(d1, d2); }
float opS(float d1, float d2) { return max(-d1, d2); }
float opI(float d1, float d2) { return max(d1, d2); }

// Smooth operations (k = smoothness, try 0.1-0.5)
float opSU(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSS(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float opSI(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// Round (add thickness)
float opRound(float d, float r) { return d - r; }

// Onion (hollow shell)
float opOnion(float d, float t) { return abs(d) - t; }

// ============================================================================
// COLOR
// ============================================================================
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Convenience: hsv with separate args
vec3 hsv(float h, float s, float v) {
    return hsv2rgb(vec3(h, s, v));
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// IQ's cosine palette: http://iquilezles.org/articles/palettes/
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

// Preset palettes
vec3 rainbow(float t) {
    return hsv(t, 0.8, 0.9);
}

vec3 fire(float t) {
    return pal(t, vec3(0.5), vec3(0.5), vec3(1.0, 0.7, 0.4), vec3(0.0, 0.15, 0.2));
}

vec3 ice(float t) {
    return pal(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 1.0), vec3(0.3, 0.2, 0.2));
}

vec3 neon(float t) {
    return pal(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.33, 0.67));
}

vec3 sunset(float t) {
    return pal(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 0.7, 0.4), vec3(0.0, 0.15, 0.2));
}

// Gamma correction
vec3 gamma(vec3 c, float g) {
    return pow(c, vec3(1.0 / g));
}

vec3 gamma(vec3 c) {
    return gamma(c, 2.2);
}

// ============================================================================
// ANIMATION & EASING
// ============================================================================
float easeIn(float t) { return t * t; }
float easeOut(float t) { return 1.0 - (1.0 - t) * (1.0 - t); }
float easeIO(float t) { return t < 0.5 ? 2.0 * t * t : 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0; }

float easeInCubic(float t) { return t * t * t; }
float easeOutCubic(float t) { return 1.0 - pow(1.0 - t, 3.0); }
float easeIOCubic(float t) { return t < 0.5 ? 4.0 * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0; }

float easeInElastic(float t) {
    return t == 0.0 ? 0.0 : t == 1.0 ? 1.0 : -pow(2.0, 10.0 * t - 10.0) * sin((t * 10.0 - 10.75) * TAU / 3.0);
}

float easeOutElastic(float t) {
    return t == 0.0 ? 0.0 : t == 1.0 ? 1.0 : pow(2.0, -10.0 * t) * sin((t * 10.0 - 0.75) * TAU / 3.0) + 1.0;
}

float easeOutBounce(float t) {
    if (t < 1.0 / 2.75) return 7.5625 * t * t;
    else if (t < 2.0 / 2.75) { t -= 1.5 / 2.75; return 7.5625 * t * t + 0.75; }
    else if (t < 2.5 / 2.75) { t -= 2.25 / 2.75; return 7.5625 * t * t + 0.9375; }
    else { t -= 2.625 / 2.75; return 7.5625 * t * t + 0.984375; }
}

// Oscillators (0-1 output)
float osc(float t) { return sin(t * TAU) * 0.5 + 0.5; }
float saw(float t) { return fract(t); }
float sqr(float t) { return step(0.5, fract(t)); }
float tri(float t) { return abs(2.0 * fract(t) - 1.0); }

// Pulse with adjustable width
float pulse(float t, float w) { return step(1.0 - w, fract(t)); }

// ============================================================================
// RAYMARCHING HELPERS
// ============================================================================
vec3 rayDir(vec2 uv, vec3 ro, vec3 ta, float zoom) {
    vec3 f = normalize(ta - ro);
    vec3 r = normalize(cross(vec3(0.0, 1.0, 0.0), f));
    vec3 u = cross(f, r);
    return normalize(uv.x * r + uv.y * u + zoom * f);
}

// Get normal using gradient (requires map() to be defined)
// Usage: vec3 n = norm(p, 0.001);
#define NORM(p, eps, map) normalize(vec3( \
    map(p + vec3(eps, 0, 0)) - map(p - vec3(eps, 0, 0)), \
    map(p + vec3(0, eps, 0)) - map(p - vec3(0, eps, 0)), \
    map(p + vec3(0, 0, eps)) - map(p - vec3(0, 0, eps)) \
))

// Simple soft shadow
float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 64; i++) {
        // Note: requires map() to be defined
        // float h = map(ro + rd * t);
        // res = min(res, k * h / t);
        // t += clamp(h, 0.02, 0.1);
        // if (res < 0.001 || t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Ambient occlusion
float ao(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i);
        // Note: requires map() to be defined
        // float d = map(p + h * n);
        // occ += (h - d) * sca;
        // sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// ============================================================================
// SCREEN / UV HELPERS
// ============================================================================
vec2 uvCenter(vec2 fragCoord, vec2 resolution) {
    return (fragCoord - 0.5 * resolution) / resolution.y;
}

vec2 uvNorm(vec2 fragCoord, vec2 resolution) {
    return fragCoord / resolution;
}

// Aspect-correct UV
vec2 uvAspect(vec2 fragCoord, vec2 resolution) {
    vec2 uv = fragCoord / resolution;
    uv.x *= resolution.x / resolution.y;
    return uv;
}

// Vignette effect
float vignette(vec2 uv, float intensity) {
    uv = uv * 2.0 - 1.0;
    return 1.0 - dot(uv, uv) * intensity;
}

// ============================================================================
// BLEND MODES
// ============================================================================
vec3 blendAdd(vec3 base, vec3 blend) { return min(base + blend, 1.0); }
vec3 blendMultiply(vec3 base, vec3 blend) { return base * blend; }
vec3 blendScreen(vec3 base, vec3 blend) { return 1.0 - (1.0 - base) * (1.0 - blend); }
vec3 blendOverlay(vec3 base, vec3 blend) {
    return mix(2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend), step(0.5, base));
}

// ============================================================================
// ADVANCED NOISE - Simplex, Curl, Worley
// ============================================================================

// Simplex 2D noise
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float simplex(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Curl noise for fluid-like motion
vec2 curl(vec2 p, float t) {
    float eps = 0.01;
    float n1 = simplex(p + vec2(eps, 0.0) + t);
    float n2 = simplex(p - vec2(eps, 0.0) + t);
    float n3 = simplex(p + vec2(0.0, eps) + t);
    float n4 = simplex(p - vec2(0.0, eps) + t);
    float dx = (n1 - n2) / (2.0 * eps);
    float dy = (n3 - n4) / (2.0 * eps);
    return vec2(dy, -dx);
}

// 3D curl noise
vec3 curl3(vec3 p) {
    float eps = 0.01;
    vec3 dx = vec3(eps, 0.0, 0.0);
    vec3 dy = vec3(0.0, eps, 0.0);
    vec3 dz = vec3(0.0, 0.0, eps);

    float x1 = noise3(p + dy) - noise3(p - dy);
    float x2 = noise3(p + dz) - noise3(p - dz);
    float y1 = noise3(p + dz) - noise3(p - dz);
    float y2 = noise3(p + dx) - noise3(p - dx);
    float z1 = noise3(p + dx) - noise3(p - dx);
    float z2 = noise3(p + dy) - noise3(p - dy);

    return vec3(x1 - x2, y1 - y2, z1 - z2) / (2.0 * eps);
}

// Worley/cellular noise
float worley(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    float minDist = 1.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);
            minDist = min(minDist, d);
        }
    }
    return sqrt(minDist);
}

// Worley with 2nd closest distance (for edges)
vec2 worley2(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    float d1 = 1.0, d2 = 1.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
    }
    return vec2(sqrt(d1), sqrt(d2));
}

// ============================================================================
// DOMAIN WARPING & DISTORTION
// ============================================================================

// Domain warp using fbm
vec2 warp(vec2 p, float amount, float t) {
    vec2 q = vec2(fbm(p + vec2(0.0, 0.0) + t * 0.1),
                  fbm(p + vec2(5.2, 1.3) + t * 0.1));
    return p + amount * q;
}

// Double domain warp (more complex patterns)
vec2 warp2(vec2 p, float amount, float t) {
    vec2 q = vec2(fbm(p + vec2(0.0, 0.0)),
                  fbm(p + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2) + t * 0.15),
                  fbm(p + 4.0 * q + vec2(8.3, 2.8) + t * 0.15));
    return p + amount * r;
}

// Swirl distortion
vec2 swirl(vec2 p, vec2 center, float amount, float radius) {
    vec2 d = p - center;
    float dist = length(d);
    float angle = amount * smoothstep(radius, 0.0, dist);
    return center + rot2(d, angle);
}

// Pinch/bulge distortion
vec2 pinch(vec2 p, vec2 center, float amount, float radius) {
    vec2 d = p - center;
    float dist = length(d);
    float factor = pow(dist / radius, amount);
    return center + normalize(d) * factor * radius * step(dist, radius) + d * step(radius, dist);
}

// Ripple distortion
vec2 ripple(vec2 p, vec2 center, float freq, float amp, float t) {
    vec2 d = p - center;
    float dist = length(d);
    float offset = sin(dist * freq - t) * amp;
    return p + normalize(d) * offset;
}

// ============================================================================
// FLUID & LIQUID EFFECTS
// ============================================================================

// Liquid surface (good for water/oil)
float liquid(vec2 p, float t, float scale) {
    float v = 0.0;
    v += sin((p.x + t) * scale);
    v += sin((p.y + t) * scale * 1.1 + 1.0);
    v += sin((p.x + p.y + t) * scale * 0.7 + 2.0);
    v += sin(length(p * scale) + t);
    return v * 0.25 + 0.5;
}

// Caustics (underwater light patterns)
float caustics(vec2 p, float t) {
    float v = 0.0;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 q = p * (1.0 + fi * 0.5);
        q += vec2(sin(t * 0.3 + fi), cos(t * 0.4 + fi));
        v += sin(q.x * 3.0 + sin(q.y * 4.0 + t)) *
             sin(q.y * 3.0 + sin(q.x * 4.0 + t * 1.1));
    }
    return v * 0.33 + 0.5;
}

// Smoke/fog density
float smoke(vec2 p, float t) {
    vec2 q = warp(p * 2.0, 0.5, t);
    float f = fbm(q * 3.0 + t * 0.3);
    f = f * f * 2.0;
    return f;
}

// ============================================================================
// GLOW & BLOOM EFFECTS
// ============================================================================

// Simple glow around a distance field
float glow(float d, float radius, float intensity) {
    return intensity / (d * d / radius + 1.0);
}

// Exponential glow
float glowExp(float d, float radius, float intensity) {
    return intensity * exp(-d * d / radius);
}

// Neon glow (sharp edge + soft glow)
vec3 neonGlow(float d, vec3 color, float sharpness, float glowSize) {
    float sharp = smoothstep(0.02, 0.0, abs(d)) * sharpness;
    float soft = exp(-abs(d) * glowSize);
    return color * (sharp + soft * 0.5);
}

// ============================================================================
// PARTICLE & SPARKLE EFFECTS
// ============================================================================

// Random sparkles
float sparkle(vec2 p, float t, float density) {
    vec2 id = floor(p * density);
    vec2 f = fract(p * density);
    float phase = hash(id) * TAU + t * (0.5 + hash(id + 100.0) * 2.0);
    float size = hash(id + 50.0) * 0.3 + 0.1;
    float d = length(f - 0.5);
    float brightness = sin(phase) * 0.5 + 0.5;
    return smoothstep(size, 0.0, d) * brightness;
}

// Star burst
float starBurst(vec2 p, int rays, float sharpness) {
    float angle = atan(p.y, p.x);
    float r = length(p);
    float ray = cos(angle * float(rays)) * 0.5 + 0.5;
    ray = pow(ray, sharpness);
    return ray * exp(-r * 3.0);
}

// Particle field (many moving dots)
float particleField(vec2 p, float t, float count, float size) {
    float v = 0.0;
    for (float i = 0.0; i < 20.0; i++) {
        if (i >= count) break;
        vec2 pos = vec2(
            sin(i * 1.3 + t * (0.3 + hash(vec2(i, 0.0)) * 0.5)),
            cos(i * 1.7 + t * (0.4 + hash(vec2(0.0, i)) * 0.5))
        ) * 0.8;
        float d = length(p - pos);
        v += smoothstep(size, 0.0, d);
    }
    return v;
}

// ============================================================================
// ELECTRIC & LIGHTNING EFFECTS
// ============================================================================

// Electric arc between two points
float electricArc(vec2 p, vec2 a, vec2 b, float t, float jitter) {
    vec2 ab = b - a;
    float len = length(ab);
    vec2 dir = ab / len;
    vec2 perp = vec2(-dir.y, dir.x);

    float proj = dot(p - a, dir);
    float dist = dot(p - a, perp);

    if (proj < 0.0 || proj > len) return 0.0;

    // Add noise displacement
    float noiseVal = simplex(vec2(proj * 10.0, t * 5.0)) * jitter;
    noiseVal += simplex(vec2(proj * 20.0, t * 8.0)) * jitter * 0.5;

    float thickness = 0.02 * (1.0 - abs(proj / len - 0.5) * 2.0);
    return smoothstep(thickness, 0.0, abs(dist - noiseVal));
}

// Plasma tendrils
float plasmaTendril(vec2 p, float t) {
    float v = 0.0;
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float angle = fi * TAU / 5.0 + t * 0.5;
        vec2 dir = vec2(cos(angle), sin(angle));
        float wave = sin(dot(p, dir) * 10.0 + t * 3.0 + fi);
        v += pow(abs(wave), 8.0);
    }
    return v;
}

// ============================================================================
// ORGANIC & CELLULAR PATTERNS
// ============================================================================

// Cell membrane look
float cells(vec2 p, float scale) {
    vec2 w = worley2(p * scale);
    return smoothstep(0.0, 0.1, w.y - w.x);
}

// Veins/cracks pattern
float veins(vec2 p, float scale) {
    vec2 w = worley2(p * scale);
    float edge = w.y - w.x;
    return 1.0 - smoothstep(0.0, 0.15, edge);
}

// Organic blobs
float organicBlob(vec2 p, float t) {
    float r = length(p);
    float angle = atan(p.y, p.x);
    float wobble = 0.0;
    for (int i = 1; i < 6; i++) {
        float fi = float(i);
        wobble += sin(angle * fi + t * (0.5 + fi * 0.1)) * (0.1 / fi);
    }
    return smoothstep(0.5 + wobble, 0.48 + wobble, r);
}

// ============================================================================
// GEOMETRIC PATTERNS
// ============================================================================

// Grid lines
float grid(vec2 p, float size, float thickness) {
    vec2 g = abs(fract(p / size - 0.5) - 0.5) * size;
    return 1.0 - smoothstep(0.0, thickness, min(g.x, g.y));
}

// Hexagonal grid
float hexGrid(vec2 p, float scale) {
    p *= scale;
    vec2 r = vec2(1.0, 1.73205);
    vec2 h = r * 0.5;
    vec2 a = mod(p, r) - h;
    vec2 b = mod(p - h, r) - h;
    vec2 gv = length(a) < length(b) ? a : b;
    return length(gv);
}

// Truchet tiles
float truchet(vec2 p, float scale) {
    vec2 id = floor(p * scale);
    vec2 f = fract(p * scale);
    float r = hash(id);
    if (r > 0.5) f = 1.0 - f;
    float d1 = length(f);
    float d2 = length(f - 1.0);
    return min(abs(d1 - 0.5), abs(d2 - 0.5));
}

// ============================================================================
// LIGHTING HELPERS
// ============================================================================

// Phong lighting model
vec3 phong(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 diffuseCol, vec3 specCol, float shininess) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return diffuseCol * diff + specCol * spec;
}

// Fresnel effect
float fresnel(vec3 normal, vec3 viewDir, float power) {
    return pow(1.0 - max(dot(normal, viewDir), 0.0), power);
}

// Rim lighting
vec3 rimLight(vec3 normal, vec3 viewDir, vec3 color, float power, float intensity) {
    float rim = pow(1.0 - max(dot(normal, viewDir), 0.0), power);
    return color * rim * intensity;
}

// ============================================================================
// POST-PROCESSING
// ============================================================================

// Chromatic aberration
vec3 chromaAb(vec2 uv, vec2 center, float amount) {
    vec2 dir = uv - center;
    // Note: requires texture sampling, placeholder
    return vec3(0.0);
}

// Film grain
float grain(vec2 p, float t, float intensity) {
    return (hash(p + t) - 0.5) * intensity;
}

// Scanlines
float scanlines(vec2 p, float count, float intensity) {
    return 1.0 - sin(p.y * count * PI) * intensity;
}

// CRT curvature
vec2 crtCurve(vec2 uv, float amount) {
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs(uv.yx) / vec2(amount);
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

// ============================================================================
// BEAT/AUDIO SYNC HELPERS
// ============================================================================

// Pulse that decays (for beat response)
float beatPulse(float trigger, float decay) {
    return trigger * exp(-decay);
}

// Smooth beat response
float beatSmooth(float beat, float attack, float release) {
    return beat; // Placeholder - actual implementation needs state
}

// Frequency band visualization helper
float freqBar(float x, float bandValue, float width, float smoothing) {
    float bar = smoothstep(0.0, smoothing, bandValue - x);
    return bar;
}
