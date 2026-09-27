#version 330 core

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in vec2 v_uv_back;
out vec4 frag_color;

uniform vec3 u_camera_pos;

// Hardcoded for M1 — surfaced as uniforms now so the python side can tweak
// them, and so the M5 material system has a natural plug-in point.
uniform vec3  u_light_dir;        // direction the light travels (unit length)
uniform vec3  u_light_color;
uniform vec3  u_material_ambient;
uniform vec3  u_material_diffuse;
uniform vec3  u_material_specular;
uniform float u_material_shininess;
uniform float u_alpha;            // M5a — face opacity (1.0 opaque; <1 for X-Ray / dim)

// M7.5a — the back side's material. Culling is disabled, so both sides of a
// face reach this shader in the same draw call and gl_FrontFacing picks the
// set to shade with. Every name here must appear in
// scene_renderer._PHONG_UNIFORMS or its location is never cached and the
// value is never set (tests/test_two_sided_shading.py holds that line).
uniform vec3  u_material_ambient_back;
uniform vec3  u_material_diffuse_back;
uniform vec3  u_material_specular_back;
uniform float u_material_shininess_back;
uniform float u_alpha_back;

// M7.5b — the material's image, one sampler per side on its own texture unit.
// A sampler cannot be left "unbound" in GLSL: it always reads whatever texture
// object sits in its unit. u_has_texture is therefore what makes an untextured
// material untextured, rather than a bound-or-not test the shader cannot make.
uniform sampler2D u_texture;
uniform sampler2D u_texture_back;
uniform float u_has_texture;
uniform float u_has_texture_back;

void main() {
    bool front = gl_FrontFacing;

    // Culling is disabled, so a back-facing fragment interpolates a normal
    // pointing away from the viewer. Flipping it is a bug fix independent of
    // the two-sided material work: without it, back faces have always been
    // lit as though they faced the other way.
    vec3 N = normalize(v_world_normal);
    if (!front) N = -N;

    vec3  m_ambient   = front ? u_material_ambient   : u_material_ambient_back;
    vec3  m_diffuse   = front ? u_material_diffuse   : u_material_diffuse_back;
    vec3  m_specular  = front ? u_material_specular  : u_material_specular_back;
    float m_shininess = front ? u_material_shininess : u_material_shininess_back;
    float m_alpha     = front ? u_alpha              : u_alpha_back;

    // The image TINTS the material rather than replacing it (spec D5), so a
    // white material shows the texture unchanged and a coloured one colourises
    // it. Its alpha multiplies the material's, which is what makes a cutout
    // PNG see-through. Ambient is tinted too, so an unlit or dimly lit part of
    // a textured face still shows the image rather than the flat colour.
    //
    // The UV SET is picked by gl_FrontFacing just like the uniform sets above:
    // front and back placements are independent (spec D7, 1.4) and both sides
    // are shaded in this one pass.
    float has_tex = front ? u_has_texture : u_has_texture_back;
    if (has_tex > 0.5) {
        vec4 texel = front ? texture(u_texture, v_uv) : texture(u_texture_back, v_uv_back);
        m_ambient *= texel.rgb;
        m_diffuse *= texel.rgb;
        m_alpha   *= texel.a;
    }

    // Convention: `u_light_dir` is the direction the light *travels* (incident
    // ray pointing INTO the surface). So `L` is the incident ray, `-L` points
    // from surface to light, and `reflect(L, N)` gives the bounce direction.
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 R = reflect(L, N);  // bounce direction (away from surface)

    float diff = max(dot(N, -L), 0.0);
    float spec = pow(max(dot(R, V), 0.0), m_shininess);

    vec3 color = m_ambient
               + m_diffuse  * diff * u_light_color
               + m_specular * spec * u_light_color;

    frag_color = vec4(color, m_alpha);
}
