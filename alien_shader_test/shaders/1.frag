// Shader 1: Basic - Gradient with Animation
// A minimal example showing UV coordinates, time, and parameter control
//
// a = horizontal color shift
// b = vertical color shift  
// c = animation speed
// d = pulse intensity

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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalize coordinates to 0-1
    vec2 uv = fragCoord / iResolution.xy;
    
    // Map parameters to useful ranges
    float speed = mix(0.5, 3.0, c);
    float pulse = mix(0.0, 0.3, d);
    
    // Animate
    float t = iTime * speed;
    float wave = sin(t) * pulse;
    
    // Create color from UV + parameters
    vec3 col = vec3(
        uv.x + a + wave,
        uv.y + b,
        0.5 + 0.5 * sin(t * 0.5)
    );
    
    fragColor = vec4(col, 1.0);
}
