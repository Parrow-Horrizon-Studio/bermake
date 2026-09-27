"""Camera for the 3D viewport: position/target/up, view + projection matrices,
and orbit/pan/zoom operations driven by mouse pixel deltas.

All math is done with numpy. The C++ kernel never sees camera state in M1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Sensitivities chosen for a 1280x800-ish widget. Tunable later via M2's preferences.
_ORBIT_RADIANS_PER_PIXEL = 0.01
_PAN_WORLD_UNITS_PER_PIXEL = 0.0015  # scaled by distance to target
_ZOOM_FACTOR_PER_SCROLL_UNIT = 0.1
_ELEVATION_CLAMP = math.radians(89.0)


def _normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        return v
    return v / length


@dataclass
class Camera:
    """Perspective camera in a Z-up world."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([8.0, -8.0, 6.0], dtype=np.float32)
    )
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))

    fov_y_deg: float = 45.0
    aspect: float = 1.0
    near: float = 0.01
    far: float = 1000.0

    # --- Matrices ----------------------------------------------------------

    def view_matrix(self) -> np.ndarray:
        """Right-handed look-at matrix mapping world -> camera space."""
        forward = _normalize(self.target - self.position)  # camera looks down -Z in cam space
        right = _normalize(np.cross(forward, self.up))
        cam_up = np.cross(right, forward)

        m = np.eye(4, dtype=np.float32)
        m[0, 0:3] = right
        m[1, 0:3] = cam_up
        m[2, 0:3] = -forward
        m[0, 3] = -float(np.dot(right, self.position))
        m[1, 3] = -float(np.dot(cam_up, self.position))
        m[2, 3] = +float(np.dot(forward, self.position))
        return m

    def projection_matrix(self) -> np.ndarray:
        """Standard OpenGL right-handed perspective projection, NDC z in [-1, 1]."""
        f = 1.0 / math.tan(math.radians(self.fov_y_deg) * 0.5)
        n, fp = self.near, self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / self.aspect
        m[1, 1] = f
        m[2, 2] = (fp + n) / (n - fp)
        m[2, 3] = (2.0 * fp * n) / (n - fp)
        m[3, 2] = -1.0
        return m

    # --- Orbit / Pan / Zoom -----------------------------------------------

    def orbit(self, dx_pixels: float, dy_pixels: float) -> None:
        """Spherical orbit around `target`. dx_pixels rotates around world Z;
        dy_pixels rotates around the camera's right vector (elevation)."""
        offset = self.position - self.target
        radius = float(np.linalg.norm(offset))
        if radius < 1e-9:
            return

        # Current spherical coords (relative to target). Azimuth is in the XY plane,
        # elevation is the angle off the XY plane toward +Z.
        azimuth = math.atan2(offset[1], offset[0])
        elevation = math.asin(float(np.clip(offset[2] / radius, -1.0, 1.0)))

        azimuth -= dx_pixels * _ORBIT_RADIANS_PER_PIXEL
        elevation += dy_pixels * _ORBIT_RADIANS_PER_PIXEL
        elevation = max(-_ELEVATION_CLAMP, min(_ELEVATION_CLAMP, elevation))

        cos_e = math.cos(elevation)
        new_offset = np.array(
            [
                radius * cos_e * math.cos(azimuth),
                radius * cos_e * math.sin(azimuth),
                radius * math.sin(elevation),
            ],
            dtype=np.float32,
        )
        self.position = self.target + new_offset

    def pan(self, dx_pixels: float, dy_pixels: float) -> None:
        """Translate position and target together along the camera's right/up axes."""
        forward = _normalize(self.target - self.position)
        right = _normalize(np.cross(forward, self.up))
        cam_up = np.cross(right, forward)

        distance = float(np.linalg.norm(self.target - self.position))
        scale = _PAN_WORLD_UNITS_PER_PIXEL * distance
        delta = (-dx_pixels * right + dy_pixels * cam_up) * scale
        self.position = self.position + delta
        self.target = self.target + delta

    def zoom(self, scroll_delta: float, cursor_ndc: np.ndarray | None = None) -> None:
        """Zoom toward the cursor (if given in NDC [-1, 1]) or toward target.

        Positive scroll_delta zooms in (gets closer); negative zooms out.
        """
        offset = self.position - self.target
        if cursor_ndc is None:
            direction = -_normalize(offset)  # toward target
        else:
            # Unproject cursor NDC to a world-space ray from the camera.
            # For zoom-toward-cursor it's sufficient to move along a screen-space
            # direction that maps to a world-space ray through the cursor pixel.
            forward = _normalize(self.target - self.position)
            right = _normalize(np.cross(forward, self.up))
            cam_up = np.cross(right, forward)
            tan_half_fovy = math.tan(math.radians(self.fov_y_deg) * 0.5)
            # cursor NDC -> camera-space direction
            cam_dir = (
                forward
                + right * (cursor_ndc[0] * tan_half_fovy * self.aspect)
                + cam_up * (cursor_ndc[1] * tan_half_fovy)
            )
            direction = _normalize(cam_dir)

        distance = float(np.linalg.norm(offset))
        step = distance * _ZOOM_FACTOR_PER_SCROLL_UNIT * scroll_delta
        # Clamp so we can't fly through the target on a single tick.
        step = max(min(step, distance * 0.9), -distance * 5.0)

        delta = direction * step
        self.position = self.position + delta
        if cursor_ndc is not None:
            # Move target with position so the view direction is preserved
            # (pure zoom-toward-cursor, no rotation). The orbit pivot does
            # drift along the cursor ray as a consequence — this matches
            # SketchUp's behavior and the user's stated preference during
            # M2 visual verification.
            self.target = self.target + delta

    # --- Picking / raycast helpers ----------------------------------------

    def ray_from_screen(
        self, x_pixels: float, y_pixels: float, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a world-space ray from the camera through the cursor pixel.

        Returns `(origin, direction)` where `origin = self.position` and
        `direction` is a unit vector. Y is treated as screen-y (top-down).
        """
        w = max(int(width), 1)
        h = max(int(height), 1)
        # NDC: x in [-1, 1] left→right; y in [-1, 1] bottom→top.
        nx = (2.0 * float(x_pixels) / w) - 1.0
        ny = 1.0 - (2.0 * float(y_pixels) / h)

        forward = _normalize(self.target - self.position)
        right = _normalize(np.cross(forward, self.up))
        cam_up = np.cross(right, forward)
        tan_half_fovy = math.tan(math.radians(self.fov_y_deg) * 0.5)

        # Camera-space direction for the (nx, ny) cursor.
        cam_dir = (
            forward + right * (nx * tan_half_fovy * self.aspect) + cam_up * (ny * tan_half_fovy)
        )
        direction = _normalize(cam_dir).astype(np.float32)
        origin = self.position.astype(np.float32)
        return origin, direction

    def world_to_screen(
        self, world_xyz: np.ndarray, width: int, height: int
    ) -> tuple[float, float, float] | None:
        """Project a world point to screen pixels. Inverse of ray_from_screen.

        Returns `(sx, sy, depth)` where (sx, sy) are pixel coordinates (screen-y
        top-down) and `depth` is the positive camera-space distance in front of
        the camera (larger = farther), suitable for depth tie-breaking. Returns
        `None` if the point is at or behind the camera plane.
        """
        w = max(int(width), 1)
        h = max(int(height), 1)
        p = np.array(
            [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0],
            dtype=np.float32,
        )
        clip = self.projection_matrix() @ (self.view_matrix() @ p)
        clip_w = float(clip[3])
        if clip_w <= 1e-7:
            return None  # at or behind the camera
        ndc_x = float(clip[0]) / clip_w
        ndc_y = float(clip[1]) / clip_w
        sx = (ndc_x + 1.0) * 0.5 * w
        sy = (1.0 - ndc_y) * 0.5 * h
        return (sx, sy, clip_w)  # clip_w == -z_cam, positive in front

    def ray_intersect_ground(
        self, x_pixels: float, y_pixels: float, width: int, height: int
    ) -> np.ndarray | None:
        """Intersect the cursor ray with the Z=0 plane.

        Returns the world-space hit point as a float32 (3,) array, or `None`
        if the ray runs parallel to the plane or hits it behind the camera.
        """
        origin, direction = self.ray_from_screen(x_pixels, y_pixels, width, height)
        dz = float(direction[2])
        if abs(dz) < 1e-9:
            return None  # parallel
        t = -float(origin[2]) / dz
        if t <= 0.0:
            return None  # behind the camera
        hit = origin + direction * t
        hit[2] = 0.0  # snap to exact zero to defend against FP drift
        return hit.astype(np.float32)
