// Shader 7: Audio Reactive - Bars/Visualizer
// Demonstrates how to use parameters for audio visualization
// Connect audio analysis (amplitude, frequency bands) to params a-h
//
// a = bass/low frequency (amplitude)
// b = mid frequency
// c = high frequency
// d = overall amplitude
// e = hue shift
// f = bar count
// g = smoothness
// h = glow

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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    uv.y = uv.y * 2.0 - 1.0; // Center vertically
    
    // Audio params
    float bass = a;
    float mid = b;
    float high = c;
    float amp = d;
    float hueShift = e;
    float barCount = mix(8.0, 32.0, f);
    float smooth_ = mix(0.01, 0.1, g);
    float glowInt = h;
    
    // Which bar are we in?
    float barIndex = floor(uv.x * barCount);
    float barX = fract(uv.x * barCount);
    
    // Interpolate between frequency bands based on position
    float t = barIndex / barCount;
    float bandValue;
    if (t < 0.33) {
        bandValue = mix(bass, mid, t * 3.0);
    } else if (t < 0.66) {
        bandValue = mix(mid, high, (t - 0.33) * 3.0);
    } else {
        bandValue = high * (1.0 - (t - 0.66) * 1.5);
    }
    
    // Add overall amplitude influence
    bandValue = bandValue * 0.7 + amp * 0.3;
    
    // Bar height
    float barHeight = bandValue;
    
    // Draw bar (mirrored)
    float bar = smoothstep(smooth_, 0.0, abs(uv.y) - barHeight);
    float edge = smoothstep(0.4, 0.5, barX) * smoothstep(0.6, 0.5, barX);
    bar *= edge;
    
    // Glow
    float glow = exp(-abs(abs(uv.y) - barHeight) * 10.0) * glowInt * bandValue;
    
    // Color
    float hue = hueShift + t * 0.3 + amp * 0.1;
    vec3 barCol = hsv(hue, 0.7, 0.9);
    vec3 glowCol = hsv(hue + 0.1, 0.8, 1.0);
    
    vec3 col = vec3(0.02);
    col += glowCol * glow;
    col = mix(col, barCol, bar);
    
    fragColor = vec4(col, 1.0);
}
