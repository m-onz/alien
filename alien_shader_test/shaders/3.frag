// Shader 3: Pattern - Grid/Tile System
// Demonstrates tiled patterns with rotation and color
//
// a = tile count
// b = rotation speed
// c = hue
// d = pattern type (0-1 selects different patterns)
// e = line thickness

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

vec2 rot2(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 hsv(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    float tiles = mix(3.0, 15.0, a);
    float rotSpeed = mix(0.0, 2.0, b);
    float hue = c;
    float patternType = d;
    float thickness = mix(0.02, 0.15, e);
    
    float t = iTime;
    
    // Tile the space
    vec2 p = uv * tiles;
    vec2 id = floor(p);
    vec2 f = fract(p) - 0.5;
    
    // Per-tile rotation
    float cellRot = hash(id) * PI + t * rotSpeed;
    f = rot2(f, cellRot);
    
    float pattern = 0.0;
    
    if (patternType < 0.25) {
        // Diagonal lines
        pattern = smoothstep(thickness, 0.0, abs(f.x + f.y));
    } else if (patternType < 0.5) {
        // Circle
        pattern = smoothstep(thickness, 0.0, abs(length(f) - 0.3));
    } else if (patternType < 0.75) {
        // Cross
        pattern = smoothstep(thickness, 0.0, min(abs(f.x), abs(f.y)));
    } else {
        // Diamond
        pattern = smoothstep(thickness, 0.0, abs(abs(f.x) + abs(f.y) - 0.3));
    }
    
    // Color per tile
    float tileHue = hue + hash(id) * 0.2;
    vec3 col = hsv(tileHue, 0.7, 0.2 + pattern * 0.8);
    
    fragColor = vec4(col, 1.0);
}
