from __future__ import annotations

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import utils


def test_ply_writer_converts_texture_coordinates_without_mutating(tmp_path) -> None:
    cube = vedo.Cube()
    original_tcoords = cube.dataset.GetPointData().GetTCoords()
    tcoords = utils.numpy2vtk(
        np.asarray(utils.vtk2numpy(original_tcoords), dtype=np.float64)
    )
    cube.dataset.GetPointData().SetTCoords(tcoords)

    assert cube.dataset.GetPointData().GetTCoords().GetDataType() != vtki.VTK_FLOAT

    output = tmp_path / "cube.ply"
    vedo.write(cube, output)

    assert output.exists()
    assert cube.dataset.GetPointData().GetTCoords().GetDataType() != vtki.VTK_FLOAT


def test_box_texture_coordinates_are_float() -> None:
    tcoords = vedo.Cube().dataset.GetPointData().GetTCoords()

    assert tcoords.GetDataType() == vtki.VTK_FLOAT
