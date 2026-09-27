#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
