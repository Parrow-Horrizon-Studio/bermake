#version 330 core

// One fullscreen quad in clip space. The NDC position is handed to the fragment
// shader, which reconstructs a world-space view ray from it.

layout(location = 0) in vec2 in_ndc;

out vec2 v_ndc;

void main() {
    v_ndc = in_ndc;
    gl_Position = vec4(in_ndc, 0.0, 1.0);
}
