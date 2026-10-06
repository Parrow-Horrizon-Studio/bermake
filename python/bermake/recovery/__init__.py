"""Autosave and crash recovery (M7.12).

This package holds the parts that need no window, so each is testable on its
own: the recovery folder (`store`), process liveness (`liveness`) and the
startup query (`startup`). It imports no Qt widgets and no OpenGL.
"""
