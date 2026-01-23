# Alien Shader Test Rig

GLSL shader development for Pure Data + GEM. Each shader demonstrates a core technique you can learn from and build on.

## Quick Start

1. Open `alien_shader_test.pd` in Pure Data (with GEM library)
2. Click the **window** toggle to open the GEM window
3. Select a shader number to load different effects
4. Move the **a-h** sliders to control the shader

## Example Shaders

Each shader demonstrates one core GLSL concept:

```
shaders/
  1.frag         - UV coordinates, time, parameters
  2.frag         - Fractal Brownian Motion (FBM)
  3.frag         - Tiled patterns with rotation
  4.frag         - 2D Signed Distance Functions
  5.frag         - 3D raymarched scene
  6.frag         - Domain warping for fluid effects
  7.frag         - Visualizer bars from audio params
  8.frag         - Post-processing effects
```

## Helper Library

Copy functions from `_lib.frag` as needed:

```
_lib.frag       - 900+ lines of helpers (noise, SDF, color, etc.)
_header.frag    - Uniforms and main() wrapper template
```

## Backups

Your personal shader collection is in `shaders_backup/` (55 shaders).

## Parameters

All shaders use **8 generic parameters (a-h)**, normalized 0-1:

| Param | Common Use |
|-------|------------|
| a | Primary control (speed, intensity) |
| b | Secondary control (size, scale) |
| c | Color/hue |
| d | Detail/complexity |
| e | Camera/depth |
| f-h | Shader-specific |

Each shader documents what its params do in the comments at the top.

## Creating New Shaders

1. Copy `_header.frag` content or use this template:

```glsl
// Shader N: My Shader Name
// a = description
// b = description
// ...

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

// Your shader code here
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Use params like: mix(0.0, 1.0, a)
    float speed = mix(0.5, 3.0, a);

    vec3 col = vec3(uv, 0.5 + 0.5 * sin(iTime * speed));
    fragColor = vec4(col, 1.0);
}
```

2. Save as `shaders/N.frag` (where N is your shader number)
3. Select that number in `_main.pd`

## Helper Library (_lib.frag)

Copy functions from `_lib.frag` into your shader as needed:

### Primitives (SDF)
- `sdSphere(p, r)`, `sdBox(p, b)`, `sdTorus(p, t)`
- `sdCircle(p, r)`, `sdRect(p, b)`, `sdHex(p, r)`

### Transforms
- `rot2(p, angle)` - 2D rotation
- `rotX/Y/Z(p, angle)` - 3D rotations
- `rep(p, c)` - infinite repetition
- `repLim(p, c, l)` - limited repetition

### SDF Operations
- `opU(a, b)` - union
- `opS(a, b)` - subtract
- `opSU(a, b, k)` - smooth union

### Color
- `hsv(h, s, v)` - HSV to RGB
- `pal(t, a, b, c, d)` - cosine palette
- `rainbow(t)`, `fire(t)`, `neon(t)`

### Noise
- `noise(p)`, `noise3(p)` - value noise
- `fbm(p)`, `fbm3(p)` - fractal brownian motion
- `voronoi(p)` - voronoi cells

### Animation
- `easeIn/Out/IO(t)` - easing functions
- `osc(t)`, `saw(t)`, `sqr(t)`, `tri(t)` - oscillators

## Connecting to Audio

Wire audio analysis from your main patch to the parameter inlets:

```
[env~] -> [/ 100] -> send to 'a' (amplitude)
[sigmund~] -> frequency bands -> 'b', 'c', 'd'
```

Or use pattern-driven values from your alien sequences.

## Tips

- Params are 0-1, use `mix(min, max, param)` to map to useful ranges
- Unused params are ignored - no errors
- Time runs automatically when toggle is on
- Mouse position available via `iMouse.xy`
