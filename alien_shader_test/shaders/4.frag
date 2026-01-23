// Shader 4: SDF - Signed Distance Functions (2D)
// Demonstrates shape rendering with SDFs
//
// a = shape morph (circle -> square -> triangle)
// b = size
// c = hue
// d = glow intensity
// e = edge sharpness

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

vec3 hsv(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

// SDF primitives
float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    float morph = a * 2.0;
    float size = mix(0.1, 0.4, b);
    float hue = c;
    float glowInt = mix(0.0, 1.0, d);
    float sharpness = mix(0.01, 0.1, e);
    
    // Animate position
    float t = iTime;
    uv += vec2(sin(t * 0.5), cos(t * 0.7)) * 0.1;
    
    // Calculate distances to each shape
    float dCircle = sdCircle(uv, size);
    float dBox = sdBox(uv, vec2(size * 0.8));
    float dTri = sdTriangle(uv, size);
    
    // Morph between shapes
    float d;
    if (morph < 1.0) {
        d = mix(dCircle, dBox, morph);
    } else {
        d = mix(dBox, dTri, morph - 1.0);
    }
    
    // Render
    float shape = smoothstep(sharpness, 0.0, d);
    float glow = exp(-abs(d) * 10.0) * glowInt;
    float edge = smoothstep(sharpness, 0.0, abs(d) - 0.01);
    
    vec3 shapeCol = hsv(hue, 0.7, 0.9);
    vec3 glowCol = hsv(hue + 0.1, 0.8, 1.0);
    vec3 bgCol = vec3(0.02);
    
    vec3 col = bgCol;
    col += glowCol * glow;
    col = mix(col, shapeCol, shape);
    col += vec3(1.0) * edge * 0.3;
    
    fragColor = vec4(col, 1.0);
}
