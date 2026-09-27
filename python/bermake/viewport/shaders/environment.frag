#version 330 core

// Sky above the horizon, ground below it, background where a half is disabled.
//
// The split is on the view ray's Z component, not on a screen-space line: the
// application is Z-up, so the ground plane is z == 0 and the horizon is exactly
// where a ray stops descending. Deriving it this way makes it correct under
// perspective and correct when the camera rolls, with no horizon geometry.

in vec2 v_ndc;

out vec4 frag_color;

uniform mat4 u_inv_view_proj;
uniform vec3 u_camera_pos;
uniform vec3 u_background;
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;
uniform float u_ground_opacity;
uniform int u_sky_enabled;
uniform int u_ground_enabled;

void main() {
    // Unproject the far plane at this fragment, then aim a ray at it.
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 world = far.xyz / far.w;
    vec3 ray = normalize(world - u_camera_pos);

    vec3 color = u_background;
    if (ray.z > 0.0 && u_sky_enabled != 0) {
        color = u_sky_color;
    } else if (ray.z < 0.0 && u_ground_enabled != 0) {
        color = mix(u_background, u_ground_color, u_ground_opacity);
    }
    frag_color = vec4(color, 1.0);
}
