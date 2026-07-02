#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Analytical/derived shapes extracted from vedo.shapes."""

import numpy as np
import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo.mesh import Mesh
from vedo.pointcloud import Points


def _convex_hull_3d_from_scipy(pts):
    try:
        from scipy.spatial import ConvexHull as ScipyConvexHull
        from scipy.spatial import QhullError
    except ImportError:
        return None

    try:
        hull = ScipyConvexHull(pts)
    except QhullError:
        return None

    faces = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        face = [int(i) for i in simplex]
        tri = pts[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.dot(normal, equation[:3]) < 0:
            face[1], face[2] = face[2], face[1]
        faces.append(face)

    if not faces:
        return None

    used = np.unique(np.asarray(faces).ravel())
    index_map = {old: new for new, old in enumerate(used)}
    compact_faces = [[index_map[i] for i in face] for face in faces]
    return Mesh([pts[used], compact_faces])


class ConvexHull(Mesh):
    """
    Create the 2D/3D convex hull from a set of points.
    """

    def __init__(self, pts) -> None:
        """
        Create the 2D/3D convex hull from a set of input points or input Mesh.

        Examples:
            - [convex_hull.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/convex_hull.py)

                ![](https://vedo.embl.es/images/advanced/convexHull.png)
        """
        if utils.is_sequence(pts):
            pts = utils.make3d(pts).astype(float)
            mesh = Points(pts)
        else:
            mesh = pts
        apoly = mesh.clean().dataset
        cpts = utils.vtk2numpy(apoly.GetPoints().GetData())

        # Create the convex hull of the pointcloud
        z0, z1 = mesh.zbounds()
        d = mesh.diagonal_size()
        if (z1 - z0) / d > 0.0001:
            hull_mesh = _convex_hull_3d_from_scipy(cpts)
            if hull_mesh is not None:
                out = hull_mesh.dataset
            else:
                delaunay = vtki.new("Delaunay3D")
                delaunay.SetInputData(apoly)
                delaunay.Update()
                surfaceFilter = vtki.new("DataSetSurfaceFilter")
                surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
                surfaceFilter.Update()
                out = surfaceFilter.GetOutput()
        else:
            delaunay = vtki.new("Delaunay2D")
            delaunay.SetInputData(apoly)
            delaunay.Update()
            fe = vtki.new("FeatureEdges")
            fe.SetInputConnection(delaunay.GetOutputPort())
            fe.BoundaryEdgesOn()
            fe.Update()
            out = fe.GetOutput()

        super().__init__(out, c=mesh.color(), alpha=0.75)
        self.flat()
        self.name = "ConvexHull"
