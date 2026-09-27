"""Viewport display-style state + descriptor table (M5a).

Pure Python — no GL imports — so it is fully unit-testable headlessly. The
renderer reads RenderStyle each frame and resolves it (see resolve_face_pass,
added in a later task) into concrete face-pass parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class FaceStyle(Enum):
    """Mutually-exclusive face display styles (the View > Face Style radio set)."""

    WIREFRAME = auto()
    HIDDEN_LINE = auto()
    MONOCHROME = auto()
    SHADED = auto()


class FaceShading(Enum):
    """How a drawn face pass is shaded."""

    LIT = auto()  # phong lighting with the (default) material color → Shaded
    UNIFORM = auto()  # phong lighting, but one fixed monochrome color    → Monochrome
    FLAT_BG = auto()  # unlit; filled with the background color           → Hidden Line


@dataclass(frozen=True)
class FaceStyleDescriptor:
    draw_faces: bool
    shading: FaceShading | None  # None iff draw_faces is False


@dataclass
class RenderStyle:
    """The viewport's current display style. One global setting per window."""

    face_style: FaceStyle = FaceStyle.SHADED
    xray: bool = False
    # M7.5a Task 11: when on, every face resolves from its owning instance's
    # tag colour instead of from its painted materials, so material COLOUR is
    # bypassed on both sides. Material opacity is not: a translucent material
    # keeps its alpha and still blends in the sorted pass. Resolved in
    # scene_renderer.py (not here), since tags are per-instance and this
    # dataclass has no scene-graph access.
    color_by_tag: bool = False


FACE_STYLE_TABLE: dict[FaceStyle, FaceStyleDescriptor] = {
    FaceStyle.WIREFRAME: FaceStyleDescriptor(draw_faces=False, shading=None),
    FaceStyle.HIDDEN_LINE: FaceStyleDescriptor(draw_faces=True, shading=FaceShading.FLAT_BG),
    FaceStyle.MONOCHROME: FaceStyleDescriptor(draw_faces=True, shading=FaceShading.UNIFORM),
    FaceStyle.SHADED: FaceStyleDescriptor(draw_faces=True, shading=FaceShading.LIT),
}

# Tunable look constants (revisited in the Task 7 visual pass).
MONO_COLOR = (0.72, 0.72, 0.74)  # uniform diffuse gray for Monochrome
XRAY_ALPHA = 0.35  # face opacity when X-Ray is on


@dataclass(frozen=True)
class PhongMaterial:
    """A phong material's color terms (ambient/diffuse/specular/shininess)."""

    ambient: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]
    shininess: float


@dataclass(frozen=True)
class FaceUniforms:
    """Resolved per-draw-call phong material uniforms + alpha, ready for the shader."""

    ambient: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]
    shininess: float
    alpha: float


def face_uniforms(
    shading: FaceShading,
    *,
    bg: tuple[float, float, float],
    material: PhongMaterial,
    xray: bool,
    material_alpha: float = 1.0,
) -> FaceUniforms:
    """Map a shading mode to concrete phong material uniforms + alpha.

    LIT     → the material's own colors (today's Shaded look).
    UNIFORM → material ambient/specular, but MONO_COLOR diffuse (Monochrome).
    FLAT_BG → unlit fill: ambient = bg, diffuse/specular = 0 (Hidden Line).
    X-Ray   → alpha = XRAY_ALPHA (else 1.0), orthogonal to shading.
    material_alpha → the material's own opacity; multiplies into the result
                      so a translucent material and X-Ray compose rather than
                      one winning outright.
    """
    alpha = (XRAY_ALPHA if xray else 1.0) * float(material_alpha)
    if shading is FaceShading.LIT:
        return FaceUniforms(
            material.ambient, material.diffuse, material.specular, material.shininess, alpha
        )
    if shading is FaceShading.UNIFORM:
        return FaceUniforms(
            material.ambient, MONO_COLOR, material.specular, material.shininess, alpha
        )
    # FaceShading.FLAT_BG
    return FaceUniforms(bg, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), material.shininess, alpha)


@dataclass(frozen=True)
class ResolvedFacePass:
    """Everything the renderer needs to draw (or skip) one definition's faces."""

    draw_faces: bool
    ambient: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]
    shininess: float
    alpha: float
    blend: bool  # enable SRC_ALPHA blending (alpha < 1.0)
    depth_write: bool  # False ⇒ glDepthMask(GL_FALSE) (X-Ray only)


def resolve_face_pass(
    style: RenderStyle,
    *,
    dimmed: bool,
    bg: tuple[float, float, float],
    material: PhongMaterial,
    dim_ambient: tuple[float, float, float],
    dim_diffuse: tuple[float, float, float],
    dim_alpha: float,
    dim_alpha_floor: float,
    material_alpha: float = 1.0,
) -> ResolvedFacePass:
    """Compose face style + X-Ray + the M4e dim pass into one face-pass result.

    Dim overrides ambient/diffuse to the desaturated dim colors (preserving the
    M4e look at the Shaded default) and multiplies alpha by dim_alpha, floored
    at dim_alpha_floor; X-Ray sets alpha to XRAY_ALPHA and turns depth writes
    off so geometry behind shows through. A translucent material_alpha behaves
    like X-Ray for depth purposes: it also stops writing depth, since blended
    geometry drawn back-to-front should not occlude what is behind it.
    """
    desc = FACE_STYLE_TABLE[style.face_style]
    if not desc.draw_faces:
        return ResolvedFacePass(
            draw_faces=False,
            ambient=(0.0, 0.0, 0.0),
            diffuse=(0.0, 0.0, 0.0),
            specular=(0.0, 0.0, 0.0),
            shininess=1.0,
            alpha=1.0,
            blend=False,
            depth_write=True,
        )
    fu = face_uniforms(
        desc.shading, bg=bg, material=material, xray=style.xray, material_alpha=material_alpha
    )
    if dimmed:
        # M7.5b (#107, refined): a floored product, not a plain product and
        # not a cap. A plain product (fu.alpha * dim_alpha) turned a 0.4
        # material into an effective 0.14 once dimmed -- very nearly
        # invisible, which is the bug #107 fixed. A plain cap
        # (min(fu.alpha, dim_alpha)) fixed that, but it also silently stopped
        # dim and X-Ray from compounding: XRAY_ALPHA is exactly dim_alpha
        # (0.35), so dimmed X-Ray collapsed to the same alpha as undimmed
        # X-Ray and the two became visually indistinguishable. Multiplying
        # keeps "dimmed is fainter" true in every case, including X-Ray's,
        # and the floor is what stops that product from vanishing for an
        # already-translucent material.
        ambient, diffuse = dim_ambient, dim_diffuse
        alpha = max(fu.alpha * dim_alpha, dim_alpha_floor)
    else:
        ambient, diffuse, alpha = fu.ambient, fu.diffuse, fu.alpha
    return ResolvedFacePass(
        draw_faces=True,
        ambient=ambient,
        diffuse=diffuse,
        specular=fu.specular,
        shininess=fu.shininess,
        alpha=alpha,
        blend=(alpha < 1.0),
        depth_write=(not style.xray and float(material_alpha) >= 1.0),
    )


# --- M5b/M7.5a: painted-material -> phong uniforms --------------------------
# phong_material_for approximates a PBR (base_color/metallic/roughness) input
# with Blinn-Phong terms, so painted faces respond to every editable field
# instead of only hue. Duplicated math (not imported) to keep render_style
# import-free of the GL renderer and of bermake.model. Unpainted front faces
# keep using scene_renderer._DEFAULT_MATERIAL directly, unchanged, so its
# hand-tuned specular/shininess no longer need to match this derivation (see
# test_phong_material_for.py's guard for the one case that still ties them
# together).
_AMBIENT_FACTOR = 0.55
_DIELECTRIC_F0 = 0.04  # normal-incidence reflectance of a non-metal
_MIN_SHININESS = 1.0
_MAX_SHININESS = 256.0

# M7.5b (#107): roughness shifts the ambient/directional-diffuse balance.
#
# A dielectric's specular is pinned at _DIELECTRIC_F0, i.e. 10/255, so 10/255
# was the hard ceiling on how far roughness could move any pixel no matter
# what the highlight lobe did -- and a measured sweep hit exactly that
# ceiling. Roughness therefore has to move a term that is not the highlight.
# It moves the one a real renderer's environment term would move: a rough
# surface scatters incoming light in every direction, so it reads flat and
# evenly lit (more ambient, less directional), and a smooth one reads
# contrasty (less ambient, more directional).
#
# Both scales are 1.0 at roughness 0.5 BY CONSTRUCTION, so the default
# dielectric that every unpainted back face and most painted faces use keeps
# exactly its v0.7.1 ambient and diffuse.
#
# The gains are tuned against a measured render, not picked to look
# reasonable in the formula. Sweeping a lit sphere through the real renderer
# and reading pixels back (v0.7.2 report, section 3):
#
#   gains       max delta   mean |delta|   mean SIGNED delta
#   1.0 / 0.7      74           16.6           +14.7   <- brightens
#   1.0 / 1.0      74           19.8            -1.1   <- chosen
#   1.0 / 1.3      89           30.9           -16.8   <- dims
#
# Equal gains are what makes the signed mean vanish: the balance point is
# _AMBIENT_FACTOR * _ROUGH_AMBIENT_GAIN / _ROUGH_DIFFUSE_GAIN == 0.55, and
# 0.55 is very close to the mean dot(N, -L) over a lit solid's visible face
# pixels, so what one term gives up the other takes back. The slider then
# redistributes rather than dimming: over the same sweep the darkest quartile
# of face pixels rises 31.7/255 and the brightest quartile falls 24.9/255.
# Magnitude 1.0 was preferred over 1.2 because it is already a 7x improvement
# on v0.7.1's 10/255 ceiling and it does not clip a bright swatch any harder
# (White 0.92 clips an identical 65.3% of face pixels before and after).
_ROUGH_AMBIENT_GAIN = 1.0
_ROUGH_DIFFUSE_GAIN = 1.0

# The unpainted back-face colour. A renderer constant rather than a library
# entry (spec D3), so it never appears as a swatch; its only job is to make a
# reversed face obvious.
BACK_DEFAULT_COLOR = (0.45, 0.50, 0.58)


def _mix(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def phong_material_for(
    base_color: tuple[float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.5,
) -> PhongMaterial:
    """Approximate a PBR material with Blinn-Phong terms.

    diffuse  = base_color * (1 - metallic) * diffuse_scale(roughness)
    specular = mix(0.04, base_color, metallic)  dielectric F0, tinted for metals
    shininess = 256 ** (1 - roughness)          256 / 64 / 16 / 4 / 1 across
                                                 the slider, never clamped
    ambient  = base_color * _AMBIENT_FACTOR * ambient_scale(roughness)

    An approximation, not real PBR: M12's renderer will not match it pixel for
    pixel. What it buys is that every editable field changes what you see.

    Three of those four lines changed in M7.5b (#107), and all three changes
    are departures from real PBR made deliberately because this renderer has
    one directional light and no environment term:

    * ambient no longer carries a (1 - metallic) factor. In real PBR a metal
      has no diffuse lobe and its appearance comes entirely from what it
      reflects; with nothing to reflect, metallic 1.0 rendered pure black off
      the highlight. A metal's colour IS its reflectance, and ambient is the
      only stand-in for environment light here, so the metal keeps it: a red
      metal now reads dark red. Dielectrics are untouched, since the dropped
      factor was 1.0 at metallic 0.
    * roughness drives the ambient/diffuse balance (see the gain constants).
    * shininess uses 256 ** (1 - roughness) instead of the spec's stated
      Blinn-Phong equivalence 2 / roughness**4 - 2. That equivalence clamped
      to _MAX_SHININESS for every roughness below 0.28, so the first third of
      the slider did nothing; it is derived for a renderer that has an
      environment term to carry the rest of the response, which this one does
      not.
    """
    r, g, b = float(base_color[0]), float(base_color[1]), float(base_color[2])
    m = min(max(float(metallic), 0.0), 1.0)
    rough = min(max(float(roughness), 0.0), 1.0)

    kd = 1.0 - m
    # Both are exactly 1.0 at the default roughness 0.5.
    ambient_scale = 1.0 + _ROUGH_AMBIENT_GAIN * (rough - 0.5)
    diffuse_scale = 1.0 - _ROUGH_DIFFUSE_GAIN * (rough - 0.5)
    ka = _AMBIENT_FACTOR * ambient_scale
    shininess = _MAX_SHININESS ** (1.0 - rough)

    return PhongMaterial(
        ambient=(r * ka, g * ka, b * ka),
        diffuse=(r * kd * diffuse_scale, g * kd * diffuse_scale, b * kd * diffuse_scale),
        specular=(
            _mix(_DIELECTRIC_F0, r, m),
            _mix(_DIELECTRIC_F0, g, m),
            _mix(_DIELECTRIC_F0, b, m),
        ),
        shininess=shininess,
    )
