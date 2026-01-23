// Shader 6: Fluid - Flow and Smoke
// Demonstrates fluid-like motion with domain warping
//
// a = flow speed
// b = turbulence
// c = hue
// d = density
// e = warp amount

/////////////////////////start Pd Header
uniform vec3 iResolution;
uniform float iTime;
uniform float iGlobalTime;
uniform vec4 iMouse;
uniform float a, b, c, d, e, f, g, h;

void mainImage(out vec4 fragColor, in vec2 fragCoord);

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
/////////////////////////end Pd Header

#define PI 3.14159265359

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

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

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 6; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + vec2(100.0);
        a *= 0.5;
    }
    return v;
}

// Domain warp
vec2 warp(vec2 p, float amount, float t) {
    vec2 q = vec2(
        fbm(p + vec2(0.0, 0.0) + t * 0.1),
        fbm(p + vec2(5.2, 1.3) + t * 0.1)
    );
    return p + amount * q;
}

vec3 hsv(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    float speed = mix(0.1, 1.0, a);
    float turbulence = mix(2.0, 8.0, b);
    float hue = c;
    float density = mix(0.5, 2.0, d);
    float warpAmt = mix(0.2, 1.5, e);
    
    float t = iTime * speed;
    
    // Warp the coordinates
    vec2 p = warp(uv * turbulence, warpAmt, t);
    
    // Layer multiple FBM
    float f1 = fbm(p + t * 0.3);
    float f2 = fbm(p * 2.0 - t * 0.2);
    float f = f1 * f2 * density;
    
    // Color gradient
    vec3 col1 = hsv(hue, 0.7, 0.9);
    vec3 col2 = hsv(hue + 0.15, 0.5, 0.3);
    vec3 col = mix(col2, col1, f);
    
    // Add some highlights
    float highlight = pow(f1, 3.0);
    col += vec3(1.0) * highlight * 0.2;
    
    fragColor = vec4(col, 1.0);
}
