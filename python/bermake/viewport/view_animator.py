"""ViewAnimator (M7e): tweens the live Camera between two CameraStates using a
Qt QVariantAnimation (built-in easing + timing). The pose math is delegated to
the pure interpolate_pose; this shell only drives ticks and writes the camera.

Only the camera is animated. Tag visibility and render style are applied
instantly by MainWindow before start() (matching SketchUp, where geometry
visibility snaps and the camera flies). Any camera input or a new recall
cancels/retargets the running animation.
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QObject, QVariantAnimation, Signal

from bermake.views.interpolate import interpolate_pose


class ViewAnimator(QObject):
    """Animate a Camera from one CameraState to another over a fixed duration."""

    finished = Signal()
    _DURATION_MS = 700

    def __init__(self, camera, on_tick, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._on_tick = on_tick
        self._from = None
        self._to = None
        self._anim = None

    def start(self, from_state, to_state) -> None:
        """Begin (or retarget) the tween from `from_state` to `to_state`."""
        self.cancel()
        self._from = from_state
        self._to = to_state
        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(self._DURATION_MS)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim.valueChanged.connect(self._on_value)
        anim.finished.connect(self._on_finished)
        self._anim = anim
        anim.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_value(self, t) -> None:
        if self._from is None or self._to is None:
            return
        interpolate_pose(self._from, self._to, float(t)).apply_to(self._camera)
        if self._on_tick is not None:
            self._on_tick()

    def _on_finished(self) -> None:
        # Land exactly on target (guards against easing not hitting 1.0 cleanly).
        if self._to is not None:
            self._to.apply_to(self._camera)
            if self._on_tick is not None:
                self._on_tick()
        self._anim = None
        self.finished.emit()

    def cancel(self) -> None:
        """Stop any running animation, leaving the camera wherever it is."""
        if self._anim is not None:
            self._anim.stop()
            self._anim.valueChanged.disconnect(self._on_value)
            self._anim.finished.disconnect(self._on_finished)
            self._anim = None

    @property
    def is_running(self) -> bool:
        return self._anim is not None
