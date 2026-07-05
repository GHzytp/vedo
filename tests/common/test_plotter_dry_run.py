#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for Plotter in dry-run mode."""

from types import SimpleNamespace

import vedo
import vedo.vtkclasses as vtki


def test_plotter_dry_run_mode_uses_fallback_screen_size() -> None:
    previous_dry_run_mode = vedo.settings.dry_run_mode
    vedo.settings.dry_run_mode = 2

    try:
        plt = vedo.Plotter(size=(1200, 600))
        result = plt.show(vedo.Sphere(), interactive=False)
        plt.close()
    finally:
        vedo.settings.dry_run_mode = previous_dry_run_mode

    assert result is plt
    assert plt.window is not None
    assert plt.renderers


def test_plotter_image_background_is_not_treated_as_color(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr(
        vedo.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(str(message)),
    )
    monkeypatch.setattr(
        vedo,
        "Image",
        lambda _filename: SimpleNamespace(actor=vtki.vtkActor()),
    )

    plt = vedo.Plotter(N=2, bg="background.png", offscreen=True)
    vedo.VedoLogo(distance=2)
    plt.close()

    assert not any("unknown color name" in warning for warning in warnings)
