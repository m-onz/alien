// _header.frag - Pd/GEM Shader Header for Alien AV Toolkit
// This file declares uniforms and wraps mainImage() for Pure Data compatibility
//
// USAGE: Prepend this to any Shadertoy-style shader
// Your shader should define: void mainImage(out vec4 fragColor, in vec2 fragCoord)

// ============================================================================
// UNIFORMS - Sent from Pure Data
// ============================================================================

// Generic control parameters (0-1 normalized)
// Map these to whatever your shader needs
uniform float a;    // param 1
uniform float b;    // param 2
uniform float c;    // param 3
uniform float d;    // param 4
uniform float e;    // param 5
uniform float f;    // param 6
uniform float g;    // param 7
uniform float h;    // param 8

// Standard Shadertoy uniforms
uniform vec3 iResolution;           // viewport resolution (in pixels)
uniform float iTime;                // shader playback time (in seconds)
uniform float iGlobalTime;          // legacy alias for iTime
uniform float iTimeDelta;           // render time (in seconds)
uniform int iFrame;                 // shader playback frame
uniform vec4 iMouse;                // mouse pixel coords. xy: current, zw: click
uniform vec4 iDate;                 // (year, month, day, time in seconds)
uniform float iSampleRate;          // sound sample rate (i.e., 44100)

// Texture inputs (if needed)
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

uniform float iChannelTime[4];
uniform vec3 iChannelResolution[4];

// ============================================================================
// COMPATIBILITY
// ============================================================================

// Alias iTime to iGlobalTime for older shaders
#ifndef iTime
#define iTime iGlobalTime
#endif

// ============================================================================
// MAIN WRAPPER
// ============================================================================

// Forward declaration - your shader must implement this
void mainImage(out vec4 fragColor, in vec2 fragCoord);

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}

// ============================================================================
// INCLUDE LIBRARY
// ============================================================================

