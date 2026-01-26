// Shader 8: Post FX - Effects you can apply to any content
// Demonstrates common post-processing effects
// Use with framebuffer input or as overlay
//
// a = vignette intensity
// b = scanlines
// c = grain amount
// d = color shift/aberration
// e = contrast
// f = saturation
// g = brightness
// h = CRT curvature

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
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// CRT curvature
vec2 crtCurve(vec2 uv, float amount) {
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs(uv.yx) / vec2(amount);
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    
    float vignette = a;
    float scanlines = b;
    float grain = c;
    float aberration = d * 0.01;
    float contrast = mix(0.5, 1.5, e);
    float saturation = mix(0.0, 2.0, f);
    float brightness = mix(0.5, 1.5, g);
    float curvature = mix(20.0, 3.0, h);
    
    // Apply CRT curvature
    if (h > 0.01) {
        uv = crtCurve(uv, curvature);
    }
    
    // Base color - animated gradient as demo content
    float t = iTime;
    vec3 col = vec3(
        0.5 + 0.5 * sin(uv.x * 3.0 + t),
        0.5 + 0.5 * sin(uv.y * 3.0 + t * 1.1 + 1.0),
        0.5 + 0.5 * sin((uv.x + uv.y) * 2.0 + t * 0.9 + 2.0)
    );
    
    // Chromatic aberration
    if (aberration > 0.0001) {
        vec2 dir = uv - 0.5;
        col.r = col.r * 0.5 + 0.5 * (0.5 + 0.5 * sin((uv.x + aberration) * 3.0 + t));
        col.b = col.b * 0.5 + 0.5 * (0.5 + 0.5 * sin((uv.x - aberration) * 3.0 + t));
    }
    
    // Contrast
    col = (col - 0.5) * contrast + 0.5;
    
    // Saturation
    float grey = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(grey), col, saturation);
    
    // Brightness
    col *= brightness;
    
    // Scanlines
    if (scanlines > 0.01) {
        float scan = sin(fragCoord.y * 2.0) * 0.5 + 0.5;
        col *= 1.0 - scanlines * 0.3 * scan;
    }
    
    // Film grain
    if (grain > 0.01) {
        float noise = (hash(uv + fract(t)) - 0.5) * grain;
        col += noise;
    }
    
    // Vignette
    if (vignette > 0.01) {
        vec2 vig = uv * (1.0 - uv);
        float v = vig.x * vig.y * 15.0;
        v = pow(v, vignette * 0.5);
        col *= v;
    }
    
    // Clamp output
    col = clamp(col, 0.0, 1.0);
    
    // Border for CRT effect
    if (h > 0.01) {
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            col = vec3(0.0);
        }
    }
    
    fragColor = vec4(col, 1.0);
}
