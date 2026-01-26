// Shader 5: Raymarching - 3D Scene
// Demonstrates basic raymarching with a rotating shape
//
// a = rotation speed
// b = shape size
// c = hue
// d = camera distance
// e = glow

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
#define MAX_STEPS 64
#define MAX_DIST 20.0
#define SURF_DIST 0.001

vec3 hsv(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

vec3 rotY(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

vec3 rotX(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

// Box SDF
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Scene
float map(vec3 p, float size, float t) {
    p = rotY(p, t);
    p = rotX(p, t * 0.7);
    return sdBox(p, vec3(size)) - 0.05; // rounded
}

// Normal calculation
vec3 getNormal(vec3 p, float size, float t) {
    float d = map(p, size, t);
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy, size, t) - d,
        map(p + e.yxy, size, t) - d,
        map(p + e.yyx, size, t) - d
    ));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    float rotSpeed = mix(0.2, 2.0, a);
    float size = mix(0.3, 0.8, b);
    float hue = c;
    float camDist = mix(2.0, 5.0, d);
    float glowInt = mix(0.0, 1.0, e);
    
    float t = iTime * rotSpeed;
    
    // Camera
    vec3 ro = vec3(0.0, 0.0, camDist);
    vec3 rd = normalize(vec3(uv, -1.0));
    
    // Raymarch
    float dO = 0.0;
    float minDist = MAX_DIST;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * dO;
        float dS = map(p, size, t);
        minDist = min(minDist, dS);
        dO += dS;
        if (dS < SURF_DIST || dO > MAX_DIST) break;
    }
    
    vec3 col = vec3(0.02);
    
    if (dO < MAX_DIST) {
        vec3 p = ro + rd * dO;
        vec3 n = getNormal(p, size, t);
        
        // Lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(n, lightDir), 0.0);
        float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 32.0);
        
        vec3 baseCol = hsv(hue, 0.6, 0.8);
        col = baseCol * (0.2 + diff * 0.6) + vec3(1.0) * spec * 0.3;
    }
    
    // Glow
    float glow = exp(-minDist * 5.0) * glowInt;
    col += hsv(hue + 0.1, 0.7, 1.0) * glow;
    
    fragColor = vec4(col, 1.0);
}
