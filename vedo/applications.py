#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Union

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo.colors import color_map, get_color
from vedo.utils import is_sequence, lin_interpolate, mag, precision
from vedo.plotter import Plotter
from vedo.pointcloud import fit_plane, Points
from vedo.shapes import Line, Ribbon, Spline, Text2D
from vedo.pyplot import CornerHistogram, histogram
from vedo.addons import SliderWidget


__docformat__ = "google"

__doc__ = """
This module contains vedo applications which provide some *ready-to-use* funcionalities

<img src="https://vedo.embl.es/images/advanced/app_raycaster.gif" width="500">
"""

__all__ = [
    "Browser",
    "IsosurfaceBrowser",
    "FreeHandCutPlotter",
    "RayCastPlotter",
    "Slicer2DPlotter",
    "Slicer3DPlotter",
    "Slicer3DTwinPlotter",
    "MorphPlotter",
    "MorphByLandmarkPlotter",
    "SplinePlotter",
    "ImageEditor",
    "AnimationPlayer",
]


#################################
class Slicer3DPlotter(Plotter):
    """
    Generate a rendering window with slicing planes for the input Volume.
    """

    def __init__(
        self,
        volume: vedo.Volume,
        cmaps=("gist_ncar_r", "hot_r", "bone", "bone_r", "jet", "Spectral_r"),
        clamp=True,
        use_slider3d=False,
        show_histo=True,
        show_icon=True,
        draggable=False,
        at=0,
        **kwargs,
    ):
        """
        Generate a rendering window with slicing planes for the input Volume.

        Arguments:
            cmaps : (list)
                list of color maps names to cycle when clicking button
            clamp : (bool)
                clamp scalar range to reduce the effect of tails in color mapping
            use_slider3d : (bool)
                show sliders attached along the axes
            show_histo : (bool)
                show histogram on bottom left
            show_icon : (bool)
                show a small 3D rendering icon of the volume
            draggable : (bool)
                make the 3D icon draggable
            at : (int)
                subwindow number to plot to
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            - [slicer1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slicer1.py)

            <img src="https://vedo.embl.es/images/volumetric/slicer1.jpg" width="500">
        """
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)

        if len(self.renderers) > 1:
            # 2d sliders do not work with multiple renderers
            use_slider3d = True

        self.volume = volume
        box = volume.box().alpha(0.2)
        self.add(box)

        volume_axes_inset = vedo.addons.Axes(
            box,
            xtitle=" ",
            ytitle=" ",
            ztitle=" ",
            yzgrid=False,
            xlabel_size=0,
            ylabel_size=0,
            zlabel_size=0,
            tip_size=0.08,
            axes_linewidth=3,
            xline_color="dr",
            yline_color="dg",
            zline_color="db",
        )

        if show_icon:
            self.add_inset(
                volume,
                volume_axes_inset,
                pos=(0.9, 0.9),
                size=0.15,
                c="w",
                draggable=draggable,
            )

        # inits
        la, ld = 0.7, 0.3  # ambient, diffuse
        dims = volume.dimensions()
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)
            # print("scalar range clamped to range: ("
            #       + precision(rmin, 3) + ", " + precision(rmax, 3) + ")")

        self.cmap_slicer = cmaps[0]

        self.current_i = None
        self.current_j = None
        self.current_k = int(dims[2] / 2)

        self.xslice = None
        self.yslice = None
        self.zslice = None

        self.zslice = volume.zslice(self.current_k).lighting("", la, ld, 0)
        self.zslice.name = "ZSlice"
        self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self.add(self.zslice)

        self.histogram = None
        data_reduced = data
        if show_histo:
            # try to reduce the number of values to histogram
            dims = self.volume.dimensions()
            n = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
            n = min(1_000_000, n)
            if data.ndim == 1:
                data_reduced = np.random.choice(data, n)
                self.histogram = histogram(
                    data_reduced,
                    # title=volume.filename,
                    bins=20,
                    logscale=True,
                    c=self.cmap_slicer,
                    bg=ch,
                    alpha=1,
                    axes=dict(text_scale=2),
                ).clone2d(pos=[-0.925, -0.88], size=0.4)
                self.add(self.histogram)

        #################
        def slider_function_x(_widget, _event):
            i = int(self.xslider.value)
            if i == self.current_i:
                return
            self.current_i = i
            self.xslice = volume.xslice(i).lighting("", la, ld, 0)
            self.xslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.xslice.name = "XSlice"
            self.remove("XSlice")  # removes the old one
            if 0 < i < dims[0]:
                self.add(self.xslice)
            self.render()

        def slider_function_y(_widget, _event):
            j = int(self.yslider.value)
            if j == self.current_j:
                return
            self.current_j = j
            self.yslice = volume.yslice(j).lighting("", la, ld, 0)
            self.yslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.yslice.name = "YSlice"
            self.remove("YSlice")
            if 0 < j < dims[1]:
                self.add(self.yslice)
            self.render()

        def slider_function_z(_widget, _event):
            k = int(self.zslider.value)
            if k == self.current_k:
                return
            self.current_k = k
            self.zslice = volume.zslice(k).lighting("", la, ld, 0)
            self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.zslice.name = "ZSlice"
            self.remove("ZSlice")
            if 0 < k < dims[2]:
                self.add(self.zslice)
            self.render()

        if not use_slider3d:
            self.xslider = self.add_slider(
                slider_function_x,
                0,
                dims[0],
                title="",
                title_size=0.5,
                pos=[(0.8, 0.12), (0.95, 0.12)],
                show_value=False,
                c=cx,
            )
            self.yslider = self.add_slider(
                slider_function_y,
                0,
                dims[1],
                title="",
                title_size=0.5,
                pos=[(0.8, 0.08), (0.95, 0.08)],
                show_value=False,
                c=cy,
            )
            self.zslider = self.add_slider(
                slider_function_z,
                0,
                dims[2],
                title="",
                title_size=0.6,
                value=int(dims[2] / 2),
                pos=[(0.8, 0.04), (0.95, 0.04)],
                show_value=False,
                c=cz,
            )

        else:  # 3d sliders attached to the axes bounds
            bs = box.bounds()
            self.xslider = self.add_slider3d(
                slider_function_x,
                pos1=(bs[0], bs[2], bs[4]),
                pos2=(bs[1], bs[2], bs[4]),
                xmin=0,
                xmax=dims[0],
                t=box.diagonal_size() / mag(box.xbounds()) * 0.6,
                c=cx,
                show_value=False,
            )
            self.yslider = self.add_slider3d(
                slider_function_y,
                pos1=(bs[1], bs[2], bs[4]),
                pos2=(bs[1], bs[3], bs[4]),
                xmin=0,
                xmax=dims[1],
                t=box.diagonal_size() / mag(box.ybounds()) * 0.6,
                c=cy,
                show_value=False,
            )
            self.zslider = self.add_slider3d(
                slider_function_z,
                pos1=(bs[0], bs[2], bs[4]),
                pos2=(bs[0], bs[2], bs[5]),
                xmin=0,
                xmax=dims[2],
                value=int(dims[2] / 2),
                t=box.diagonal_size() / mag(box.zbounds()) * 0.6,
                c=cz,
                show_value=False,
            )

        #################
        def button_func(_obj, _evtname):
            bu.switch()
            self.cmap_slicer = bu.status()
            for m in self.objects:
                try:
                    if "Slice" in m.name:
                        m.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
                except AttributeError:
                    pass
            self.remove(self.histogram)
            if show_histo:
                self.histogram = histogram(
                    data_reduced,
                    # title=volume.filename,
                    bins=20,
                    logscale=True,
                    c=self.cmap_slicer,
                    bg=ch,
                    alpha=1,
                    axes=dict(text_scale=2),
                ).clone2d(pos=[-0.925, -0.88], size=0.4)
                self.add(self.histogram)
            self.render()

        if len(cmaps) > 1:
            bu = self.add_button(
                button_func,
                states=cmaps,
                c=["k9"] * len(cmaps),
                bc=["k1"] * len(cmaps),  # colors of states
                size=16,
                bold=True,
            )
            if bu:
                bu.pos([0.04, 0.01], "bottom-left")


####################################################################################
class Slicer3DTwinPlotter(Plotter):
    """
    Create a window with two side-by-side 3D slicers for two Volumes.

    Arguments:
        vol1 : (Volume)
            the first Volume object to be isosurfaced.
        vol2 : (Volume)
            the second Volume object to be isosurfaced.
        clamp : (bool)
            clamp scalar range to reduce the effect of tails in color mapping
        **kwargs : (dict)
            keyword arguments to pass to a `vedo.plotter.Plotter` instance.

    Example:
        ```python
        from vedo import *
        from vedo.applications import Slicer3DTwinPlotter

        vol1 = Volume(dataurl + "embryo.slc")
        vol2 = Volume(dataurl + "embryo.slc")

        plt = Slicer3DTwinPlotter(
            vol1, vol2,
            shape=(1, 2),
            sharecam=True,
            bg="white",
            bg2="lightblue",
        )

        plt.at(0).add(Text2D("Volume 1", pos="top-center"))
        plt.at(1).add(Text2D("Volume 2", pos="top-center"))

        plt.show(viewup='z')
        plt.at(0).reset_camera()
        plt.interactive().close()
        ```

        <img src="https://vedo.embl.es/images/volumetric/slicer3dtwin.png" width="650">
    """

    def __init__(self, vol1: vedo.Volume, vol2: vedo.Volume, clamp=True, **kwargs):

        super().__init__(**kwargs)

        cmap = "gist_ncar_r"
        cx, cy, cz = "dr", "dg", "db"  # slider colors
        ambient, diffuse = 0.7, 0.3  # lighting params

        self.at(0)
        box1 = vol1.box().alpha(0.1)
        box2 = vol2.box().alpha(0.1)
        self.add(box1)

        self.at(1).add(box2)
        self.add_inset(vol2, pos=(0.85, 0.15), size=0.15, c="white", draggable=0)

        dims = vol1.dimensions()
        data = vol1.pointdata[0]
        rmin, rmax = vol1.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)

        def slider_function_x(_widget, _event):
            i = int(self.xslider.value)
            msh1 = vol1.xslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "XSlice"
            self.at(0).remove("XSlice")  # removes the old one
            msh2 = vol2.xslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "XSlice"
            self.at(1).remove("XSlice")
            if 0 < i < dims[0]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        def slider_function_y(_widget, _event):
            i = int(self.yslider.value)
            msh1 = vol1.yslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "YSlice"
            self.at(0).remove("YSlice")
            msh2 = vol2.yslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "YSlice"
            self.at(1).remove("YSlice")
            if 0 < i < dims[1]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        def slider_function_z(_widget, _event):
            i = int(self.zslider.value)
            msh1 = vol1.zslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "ZSlice"
            self.at(0).remove("ZSlice")
            msh2 = vol2.zslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "ZSlice"
            self.at(1).remove("ZSlice")
            if 0 < i < dims[2]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        self.at(0)
        bs = box1.bounds()
        self.xslider = self.add_slider3d(
            slider_function_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0,
            xmax=dims[0],
            t=box1.diagonal_size() / mag(box1.xbounds()) * 0.6,
            c=cx,
            show_value=False,
        )
        self.yslider = self.add_slider3d(
            slider_function_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0,
            xmax=dims[1],
            t=box1.diagonal_size() / mag(box1.ybounds()) * 0.6,
            c=cy,
            show_value=False,
        )
        self.zslider = self.add_slider3d(
            slider_function_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0,
            xmax=dims[2],
            value=int(dims[2] / 2),
            t=box1.diagonal_size() / mag(box1.zbounds()) * 0.6,
            c=cz,
            show_value=False,
        )

        #################
        hist = CornerHistogram(data, s=0.2, bins=25, logscale=True, c="k")
        self.add(hist)
        slider_function_z(0, 0)  ## init call


########################################################################################
class MorphPlotter(Plotter):
    """
    A Plotter with 3 renderers to show the source, target and warped meshes.

    Examples:
        - [warp4b.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp4b.py)

            ![](https://vedo.embl.es/images/advanced/warp4b.jpg)
    """

    def __init__(self, source, target, **kwargs):

        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.enable_default_mouse_callbacks = False

        kwargs.update({"N": 3})
        kwargs.update({"sharecam": 0})

        super().__init__(**kwargs)

        self.n_intermediates = 10  # number of intermediate shapes to generate
        self.intermediates = []
        self.auto_warping = True   # automatically warp the source mesh on click
        self.automatic_picking_distance = 0.075
        self.automatic_picking_use_source = False # use source mesh to pick points
        self.cmap_name = "coolwarm"
        self.output_filename = "morphed.vtk"
        self.nbins = 25

        self.source = source.pickable(True)
        self.target = target.pickable(False)
        self.dottedln = None
        self.clicked = []
        self.sources = []
        self.targets = []
        self.warped = None
        self.source_labels = None
        self.target_labels = None
        self.msg0 = Text2D(
            "Pick a point on the surface",
            pos="bottom-center", c='white', bg="blue4", alpha=1, font="Calco")
        self.msg1 = Text2D(pos="bottom-center", c='white', bg="blue4", alpha=1, font="Calco")
        self.instructions = Text2D(s=0.7, bg="blue4", alpha=0.1, font="Calco")
        self.instructions.text(
            "  Morphological alignment of 3D surfaces\n"
            "Pick a point on the source surface, then\n"
            "pick the corresponding point on the target \n"
            "Pick at least 4 point pairs. Press:\n"
            "- d to delete the last landmark pair\n"
            "- C to clear all landmarks\n"
            "- f to add a pair of fixed landmarks\n"
            "- a to auto-pick additional landmarks\n"
            "- z to compute and show the residuals\n"
            "- g to generate intermediate shapes\n"
            "- Ctrl+s to save the morphed mesh\n"
            "- q to quit and proceed"
        )
        self.output_text = Text2D("Morphing Output", font="Calco")
        self.at(0).add_renderer_frame()
        self.add(source, self.msg0, self.instructions).reset_camera()
        self.at(1).add_renderer_frame()
        self.add(Text2D(f"Target: {target.filename[-35:]}", bg="blue4", alpha=0.1, font="Calco"))
        self.add(self.msg1, target)
        cam1 = self.camera  # save camera at 1
        self.at(2).background("k9")
        self.camera = cam1  # use the same camera of renderer1

        self.add(target, self.output_text)
        self.add_renderer_frame()

        self.callid1 = self.add_callback("KeyPress", self.on_keypress)
        self.callid2 = self.add_callback("LeftButtonPress", self.on_click)
        self._interactive = True

    ################################################
    def update(self):
        """Update the rendering window"""
        source_pts = Points(self.sources).color("purple5").ps(12)
        target_pts = Points(self.targets).color("purple5").ps(12)
        source_pts.name = "source_pts"
        target_pts.name = "target_pts"
        self.source_labels = source_pts.labels2d("id", c="purple3")
        self.target_labels = target_pts.labels2d("id", c="purple3")
        self.source_labels.name = "source_pts"
        self.target_labels.name = "target_pts"
        self.at(0).remove("source_pts").add(source_pts, self.source_labels)
        self.at(1).remove("target_pts").add(target_pts, self.target_labels)
        self.render()

        if self.auto_warping:
            if len(self.sources) == len(self.targets) and len(self.sources) > 3:
                self.warped = self.source.clone().warp(self.sources, self.targets)
                self.warped.name = "warped"
                self.at(2).remove("warped").add(self.warped)
                self.render()

    def on_click(self, evt):
        """Handle mouse click events"""
        if evt.object == self.source:
            self.sources.append(evt.picked3d)
            self.source.pickable(False)
            self.target.pickable(True)
            self.msg0.text("--->")
            self.msg1.text("now pick a target point")
            self.update()
        elif evt.object == self.target:
            self.targets.append(evt.picked3d)
            self.source.pickable(True)
            self.target.pickable(False)
            self.msg0.text("now pick a source point")
            self.msg1.text("<---")
            self.update()

    def on_keypress(self, evt):
        """Handle keyboard events"""
        if evt.keypress == "C":
            self.sources.clear()
            self.targets.clear()
            self.at(0).remove("source_pts")
            self.at(1).remove("target_pts")
            self.at(2).remove("warped")
            self.msg0.text("CLEARED! Pick a point here")
            self.msg1.text("")
            self.source.pickable(True)
            self.target.pickable(False)
            self.update()

        if evt.keypress == "w":
            rep = (self.warped.properties.GetRepresentation() == 1)
            self.warped.wireframe(not rep)
            self.render()

        if evt.keypress == "d":
            if len(self.sources) == len(self.targets) + 1:
                self.sources.pop()
            elif len(self.targets) == len(self.sources) + 1:
                self.targets.pop()
            elif len(self.sources) == 0 or len(self.targets) == 0:
                return
            n = min(len(self.sources), len(self.targets))
            self.sources = self.sources[:n-1]
            self.targets = self.targets[:n-1]
            self.msg0.text("Last point deleted! Pick a point here")
            self.msg1.text("")
            self.source.pickable(True)
            self.target.pickable(False)
            self.update()

        if evt.keypress == "f":
            # add a pair of fixed landmarks, a point that does not change
            if evt.object == self.source and len(self.sources) == len(self.targets):
                self.sources.append(evt.picked3d)
                self.targets.append(evt.picked3d)
                self.update()

        if evt.keypress == "m":
            if len(self.sources) == len(self.targets) and len(self.sources) > 3:
                self.warped = self.source.clone().warp(self.sources, self.targets)
                self.warped.name = "warped"
                self.output_text.text("Morphed output:")
                self.at(2).remove("warped").add(self.warped).render()

        if evt.keypress == 'g':  ##------- generate intermediate shapes
            if not self.warped:
                vedo.printc("Morph the source mesh first.", c="r")
                return
            if len(self.sources) != len(self.targets) or len(self.sources) < 4:
                vedo.printc("Pick at least 4 pairs of points.", c="r")
                return
            self.output_text.text("Generating intermediate shapes...")
            self.output_text.c("white").background("red4")
            self.arrow_starts = np.array(self.sources)
            self.arrow_stops = np.array(self.targets)
            self.dottedln = vedo.Lines(self.arrow_starts, self.arrow_stops, res=self.n_intermediates)
            self.dottedln.name = "warped"
            self.dottedln.c("blue3").ps(5).alpha(0.5)
            self.at(2).add(self.dottedln).render()
            self.intermediates = []
            allpts = self.dottedln.vertices
            allpts = allpts.reshape(len(self.arrow_starts), self.n_intermediates + 1, 3)
            for i in range(self.n_intermediates):
                pi = allpts[:, i, :]
                m_nterp = self.source.clone().warp(self.arrow_starts, pi)
                m_nterp.name = f"morphed"
                m_nterp.c("blue4", 0.05)
                self.intermediates.append(m_nterp)
            self.output_text.text(f"Morphed output + Interpolation ({self.n_intermediates} shapes):")
            self.output_text.c("k").background(None)
            self.at(2).add(self.intermediates).render()

        if evt.keypress == "a":
            # auto-pick points on the target surface
            if not self.warped:
                vedo.printc("At least 4 points are needed.", c="r")
                return
            if len(self.sources) > len(self.targets):
                self.sources.pop()

            if self.automatic_picking_use_source:
                pts = self.warped.clone().subsample(self.automatic_picking_distance)
                TI = self.warped.transform.compute_inverse()
                # d = self.target.diagonal_size()
                # r = d * self.automatic_picking_distance
                for p in pts.coordinates:
                    # pp = vedo.utils.closest(p, self.targets)[1]
                    # if vedo.mag(pp - p) < r:
                    #     continue
                    q = self.target.closest_point(p)
                    self.sources.append(TI(p))
                    self.targets.append(q)
            else:
                pts = self.target.clone().subsample(self.automatic_picking_distance)
                d = self.target.diagonal_size()
                r = d * self.automatic_picking_distance
                TI = self.warped.transform.compute_inverse()
                for p in pts.coordinates:
                    pp = vedo.utils.closest(p, self.targets)[1]
                    if vedo.mag(pp - p) < r:
                        continue
                    q = self.warped.closest_point(p)
                    self.sources.append(TI(q))
                    self.targets.append(p)

            self.source.pickable(True)
            self.target.pickable(False)
            self.update()

        if evt.keypress == "z" or evt.keypress == "a":
            dists = self.warped.distance_to(self.target, signed=True)
            v = np.std(dists) * 2
            self.warped.cmap(self.cmap_name, dists, vmin=-v, vmax=+v)

            h = vedo.pyplot.histogram(
                dists,
                bins=self.nbins,
                title=" ",
                xtitle=f"STD = {v/2:.2f}",
                ytitle="",
                c=self.cmap_name,
                xlim=(-v, v),
                aspect=16/9,
                axes=dict(
                    number_of_divisions=5,
                    text_scale=2,
                    xtitle_offset=0.075,
                    xlabel_justify="top-center",
                ),
            )

            # try to fit a gaussian to the histogram
            def gauss(x, A, B, sigma):
                return A + B * np.exp(-x**2 / (2 * sigma**2))
            try:
                from scipy.optimize import curve_fit
                inits = [0, len(dists)/self.nbins*2.5, v/2]
                popt, _ = curve_fit(gauss, xdata=h.centers, ydata=h.frequencies, p0=inits)
                x = np.linspace(-v, v, 300)
                h += vedo.pyplot.plot(x, gauss(x, *popt), like=h, lw=1, lc="k2")
                h["Axes"]["xtitle"].text(f":sigma = {abs(popt[2]):.3f}", font="VictorMono")
            except:
                pass

            h = h.clone2d(pos="bottom-left", size=0.575)
            h.name = "warped"
            self.at(2).add(h)
            self.render()
        
        if evt.keypress == "Ctrl+s":
            # write the warped mesh to file along with the transformation
            if self.warped:
                m =  self.warped.clone()
                m.pointdata.remove("Scalars")
                m.pointdata.remove("Distance")
                m.write(str(self.output_filename))
                matfile = str(self.output_filename).split(".")[0] + ".mat"
                m.transform.write(matfile)
                print(f"Warped mesh saved to: {self.output_filename}\n with transformation: {matfile}")

        if evt.keypress == "q":
            self.break_interaction()

########################################################################################
class MorphByLandmarkPlotter(Plotter):

    def __init__(self, msh, init_pts=(), show_labels=True, **kwargs):
        """
        A Plotter to morph a mesh by moving points on it.
        This application allows you to morph a mesh by moving points on it.
        It is useful in shape analysis, morphometrics, and mesh deformation.
        The initial points are used as source points, and the points are moved to the target positions.

        Arguments:
            msh : (Mesh)
                the mesh to morph
            init_pts : (list of tuples)
                initial points to use for morphing
            show_labels : (bool)
                show point labels on the mesh
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.
        
        Example:
            ```python
            from vedo import Mesh, MorphByLandmarkPlotter
            msh = Mesh(dataurl + "bunny.ply")
            pts = msh.clone().subsample(0.1) # subsample points
            plt = MorphByLandmarkPlotter(msh, pts.coordinates, axes=1)
            plt.morphing_on_change = True  # morph the mesh when points change
            plt.show()
            ```
        """
        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.enable_default_mouse_callbacks = False

        super().__init__(**kwargs)
        self.parallel_projection(True)

        self.step = 10
        self.point_size = 20
        self.morphing_mode = "2d"  # "2d", "3d"
        self.morphing_on_change = True  # morph the mesh when points change
        self.recompute_normals = False
        self.move_keys = ["Left", "Right", "Up", "Down", "UP", "DOWN", "n", "N"]
        self.output_filename = "morphed.vtk"

        self.mesh = msh.clone().rename("morphable_mesh").pickable(1)
        self.mesh.compute_normals(cells=False)
        self.selected_pid = None
        self.highlighted = None

        self.sources = list(init_pts)
        self.targets = list(init_pts)
        self._sources = np.array(self.sources, dtype=float)
        self.vhighlighted = vedo.Point(r=self.point_size + 5, c="red6", alpha=0.5)
        self.vhighlighted.off().pickable(0)

        self.current_pid = None
        self.current_pt = None
        self.id_callback1 = self.add_callback("key press", self._fkey)
        self.id_callback2 = self.add_callback("mouse click", self._on_mouse_click)

        self.vtargets = Points(self.targets, r=self.point_size, c="blue6", alpha=0.5)
        self.vtargets.name = "vtargets"

        self.vlines = vedo.Lines(self.sources, self.targets, c="k4")
        self.vlines.name = "vlines"

        self.vsources = Points(self.sources, r=self.point_size - 1)
        self.vsources.c("red6", alpha=0.5).pickable(0)
        self.vsources.name = "vsources"

        self.instructions = Text2D(
            "Click on a point to select it.\n"
            "Use the arrow keys to move the selected point.\n"
            "Press n or N to move the point along its normal.\n"
            "Press o to add a new point pair.\n"
            "Press +/- to change the step size.\n"
            "Press m to morph the mesh.\n"
            "Press r to reset the camera.\n"
            "Press Ctrl+s to save the morphed mesh.\n"
            "Press q to quit.",
            pos="top-left",
            bg="blue4",
            alpha=0.1,
            font="Calco",
            s=0.7,
        )

        self.status = Text2D(
            "No point selected.",
            pos="bottom-left",
            font="Calco",
            bg="yellow5",
            alpha=0.2,
            s=0.8,
        )

        self.labels = None
        if show_labels:
            self.labels = self.vtargets.labels2d("id")
            self.labels.name = "labels"
        self.add(
            self.mesh,
            self.labels,
            self.vsources,
            self.vtargets,
            self.vlines,
            self.vhighlighted,
            self.instructions,
            self.status,
        )

    def do_morph(self):
        """
        Perform the morphing operation on the mesh.
        This method morphs the mesh using the provided source and target points.
        """
        self.sources = list(self.vsources.coordinates)
        self.targets = list(self.vtargets.coordinates)

        self.mesh.warp(self.sources, self.targets, mode=self.morphing_mode)

        self.sources = list(self.targets)

        self.vsources = Points(self.sources, r=self.point_size - 1)
        self.vsources.c("red6", alpha=0.5).pickable(0)
        self.vsources.name = "vsources"

        self.vtargets = Points(self.targets, r=self.point_size)
        self.vtargets.c("blue6", alpha=0.5).pickable(0)
        self.vtargets.name = "vtargets"

        self.vlines = vedo.Lines(self.vsources, self.targets, c="k4").pickable(0)
        self.vlines.name = "vlines"
        self.remove("vtargets", "vlines")
        self.add(self.vtargets, self.vlines).render()

    def _on_mouse_click(self, evt):
        if evt.picked3d is None:
            return
        self.current_pid = self.vtargets.closest_point(evt.picked3d, return_point_id=True)
        if self.current_pid < 0:
            self.status.text("No point exists. Press o to place a point.")
            self.render()
            return
        self.current_pt = self.targets[self.current_pid]
        self.status.text(
            f"Selected point ID {self.current_pid}" 
            f" coordinates: {precision(self.current_pt, 3)}"
        )
        self.vhighlighted.on()
        self.vhighlighted.coordinates = [self.current_pt]
        self.render()

    def _fkey(self, evt):
        k = evt.keypress
        # print(f"Key pressed: {k}")

        if k == "q":
            self.break_interaction()

        elif k == "r":
            self.reset_camera().render()

        elif k == "o":
            if evt.object.name == "morphable_mesh":
                self.sources.append(evt.picked3d)
                if len(self._sources) == 0:
                    self._sources = np.array([evt.picked3d], dtype=float)
                else:
                    self._sources = np.append(self._sources, [evt.picked3d], axis=0)
                self.targets.append(evt.picked3d)
                self.status.text(
                    f"Added point {precision(evt.picked3d, 3)}. "
                    f" Total number of points: {len(self._sources)}"
                )

                self.vsources = Points(self._sources, r=self.point_size - 1)
                self.vsources.c("red6", alpha=0.5).pickable(0)
                self.vsources.name = "vsources"

                self.vtargets = Points(self.targets, r=self.point_size)
                self.vtargets.c("blue6", alpha=0.5).pickable(0)
                self.vtargets.name = "vtargets"

                self.vlines = vedo.Lines(self._sources, self.targets, c="k5").pickable(0)
                self.vlines.name = "vlines"

                self.remove("vsources", "vtargets", "vlines")
                self.add(self.vsources, self.vtargets, self.vlines).render()

        elif k == "c":  # clear all points
            self.sources.clear()
            self.targets.clear()
            self._sources = np.array(self.sources, dtype=float)
            self.vsources = Points(self._sources, r=self.point_size - 1)
            self.vsources.c("red6", alpha=0.5).pickable(0)
            self.vsources.name = "vsources"
            self.vtargets = Points(self.targets, r=self.point_size)
            self.vtargets.c("blue6", alpha=0.5).pickable(0)
            self.vtargets.name = "vtargets"
            self.vlines = vedo.Lines(self._sources, self.targets, c="k4").pickable(0)
            self.vlines.name = "vlines"
            self.remove("vsources", "vtargets", "vlines", "labels")
            self.add(self.vsources, self.vtargets, self.vlines).render()

        elif k == "d":  # delete the last point pair
            if len(self.sources) == 0 or len(self.targets) == 0:
                self.status.text("No points to delete.")
                return
            if len(self.sources) == len(self.targets) + 1:
                self.sources.pop()
            elif len(self.targets) == len(self.sources) + 1:
                self.targets.pop()

            self._sources = np.array(self.sources, dtype=float)
            self.vsources = Points(self._sources, r=self.point_size - 1)
            self.vsources.c("red6", alpha=0.5).pickable(0)
            self.vsources.name = "vsources"
            self.vtargets = Points(self.targets, r=self.point_size)
            self.vtargets.c("blue6", alpha=0.5).pickable(0)
            self.vtargets.name = "vtargets"
            self.vlines = vedo.Lines(self._sources, self.targets, c="k4").pickable(0)
            self.vlines.name = "vlines"
            self.remove("vsources", "vtargets", "vlines", "labels")
            self.add(self.vsources, self.vtargets, self.vlines).render()

        elif k in self.move_keys:  # use arrows to move the picked point
            if self.current_pid is None:
                self.status.text("No point picked. Click to pick a point first.")
                return

            pid = self.current_pid

            std_keys = ["Left", "Right", "Up", "Down", "UP", "DOWN", "n", "N"]
            i = self.move_keys.index(k)
            k = std_keys[i]  # normalize the key to the standard keys

            if   k == "Left":  self.targets[pid] += [-self.step, 0, 0]
            elif k == "Right": self.targets[pid] += [self.step, 0, 0]
            elif k == "Up":    self.targets[pid] += [0, self.step, 0]
            elif k == "Down":  self.targets[pid] += [0, -self.step, 0]
            elif k == "UP":    self.targets[pid] += [0, 0, self.step]
            elif k == "DOWN":  self.targets[pid] += [0, 0, -self.step]
            elif k == "n" or k == "N":
                # move along the normal of the picked point
                if self.recompute_normals:
                    self.mesh.compute_normals(cells=False)
                mid = self.mesh.closest_point(self.current_pt, return_point_id=True)
                normals = self.mesh.point_normals.copy()
                if k == "n":
                    self.targets[pid] += normals[mid] * self.step
                else:
                    self.targets[pid] -= normals[mid] * self.step

            self.status.text(
                f"Moved point ID {self.current_pid}" 
                f" {precision(self.targets[pid], 3)} {k} by {self.step} units"
            )
            self.vtargets = Points(self.targets, r=self.point_size)
            self.vtargets.c("blue6", alpha=0.5).pickable(0)
            self.vtargets.name = "vtargets"
            self.vlines = vedo.Lines(self._sources, self.targets, c="k4").pickable(0)
            self.vlines.name = "vlines"
            self.remove("vlines", "vtargets").add(self.vlines, self.vtargets)
            if self.morphing_on_change:
                self.do_morph()
            else:
                self.render()

        elif k == "m":
            self.do_morph()

        elif k == "l" and evt.object:  # toggle lines visibility
            ev = evt.object.properties.GetEdgeVisibility()
            evt.object.properties.SetEdgeVisibility(not ev)
            self.render()

        elif k in ["plus", "equal", "minus", "underscore"]:  # change step size
            if k in ["plus", "equal"]:
                self.step += self.step / 10
            elif k in ["minus", "underscore"]:
                self.step -= self.step / 10
            self.step = max(0, self.step)  # ensure step is at least 0
            self.status.text(f"Step size changed to {self.step:.2f} units.")
            self.render()

        elif k == "Ctrl+s":  # write the morphed mesh to file
            if self.mesh:
                # self.mesh.compute_normals(cells=False)
                self.mesh.write(self.output_filename)
                self.status.text(f"Morphed mesh saved to {self.output_filename}")
                self.render()

########################################################################################
class MorphBySplinesPlotter(Plotter):

    def __init__(self, meshes, **kwargs):
        """
        A Plotter to morph a mesh by moving points on it using splines.
        It is useful in shape analysis, morphometrics, and mesh deformation.

        Arguments:
            meshes : (list of Mesh)
                the mesh set to be matched
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.
        
        Example:
            ```python
            from vedo import Mesh, dataurl
            from vedo.applications import MorphBySplinesPlotter

            m0 = Mesh(dataurl+"250.vtk").subdivide().c("o8", 0.5)
            mm = Mesh(dataurl+"270.vtk").subdivide().c("k5", 0.5).x(800)
            m1 = Mesh(dataurl+"290.vtk").subdivide().c("g8", 0.5).x(2000)

            plt = MorphBySplinesPlotter([m0, mm, m1], size=(2050, 1250), axes=14)
            plt.n_intermediates = 5  # number of intermediate shapes to generate
            plt.show().close()
            ```
        """
        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.enable_default_mouse_callbacks = False
        
        super().__init__(**kwargs)

        self.parallel_projection(True)
        self.add_callback("key press", self._func)

        self.n_intermediates = 10  # number of intermediate shapes to generate
        self.mode = "2d"  # "2d", "3d" morphing mode

        self.splinetools = []
        self.intermediates = []
        self.meshes = meshes
        self.instructions = Text2D(
            "     --- Morphing by Splines ---\n"
            "Ctrl+l to load the splines from file.\n"
            "Ctrl+a to add a new spline.\n"
            "Ctrl+f to fit the splines to the mesh.\n"
            "Ctrl+z to remove the last spline.\n"
            "Ctrl+c to clear all splines.\n"
            "Ctrl+g to generate intermediate shapes.\n"
            "Ctrl+s to save the splines.\n"
            "Ctrl+q to quit.",
            pos="top-left",
            bg="blue4",
            alpha=0.1,
            font="Calco",
            s=0.75,
        )
        self.status = Text2D(
            "Use Ctrl+a to add a new spline, or Ctrl+l to load splines from file.",
            pos="bottom-left",
            font="Calco",
            bg="yellow5",
            alpha=0.5,
            s=1.1,
        )
        b = meshes[0].bounds()
        self.offset = [0, (b[3] - b[2])/2, 0]
        self.output_prefix = "morphed_"
        self.output_suffix = ".vtk"
        self.output_spline_filename = f"{self.output_prefix}splines.npy"
        self.output_offset = 0  # used to generate an offset for the output filenames
        self.output_offset_factor = 1
        self.add(meshes, self.status, self.instructions)


    def _func(self, evt):

        n = len(self.meshes)

        if evt.keypress in ["q", "Ctrl+q"]:
            self.break_interaction()

        elif evt.keypress == "Ctrl+a":
            if len(self.splinetools) > 0:
                init_nodes = self.splinetools[-1].nodes() + self.offset
            else:
                b = self.meshes[0].bounds()
                init_nodes = vedo.Line(
                    [b[0], b[2], (b[4] + b[5]) / 2],
                    [b[1], b[3], (b[4] + b[5]) / 2], res=n).coordinates
            st = self.add_spline_tool(
                init_nodes,
                lc=len(self.splinetools),
                lw=3,
                can_add_nodes=False,
            )
            self.splinetools.append(st)
            self.status.text(f"Added spline.")
            self.render()

        elif evt.keypress == "Ctrl+f":
            for st in self.splinetools:
                nodes = st.nodes()
                for i in range(len(nodes)):
                    nodes[i] = self.meshes[i].closest_point(nodes[i])
                st.set_nodes(nodes)
            self.status.text("Snapped spline points to meshes.")
            self.render()

        elif evt.keypress == "Ctrl+s":
            all_nodes = []
            for st in self.splinetools:
                all_nodes.append(st.nodes())
            all_nodes = np.array(all_nodes)
            fname = f"{self.output_prefix}splines.npy"
            np.save(fname, all_nodes)
            for i, mi in enumerate(self.intermediates):
                mi.write(f"{self.output_prefix}{self.output_offset+i*int(self.output_offset_factor)}{self.output_suffix}")
            self.status.text(
                f"Saved {len(self.splinetools)} splines to {fname}.\n"
                f"Saved {len(self.intermediates)} intermediate meshes saved as {self.output_prefix}*{self.output_suffix}")
            self.render()

        elif evt.keypress == "Ctrl+l":
            if not os.path.exists(f"{self.output_prefix}splines.npy"):
                self.status.text(f"File {self.output_prefix}splines.npy does not exist.")
                self.render()
                return
            all_nodes = np.load(f"{self.output_prefix}splines.npy")
            for st in self.splinetools:
                st.off()
            self.splinetools.clear()
            for nodes in all_nodes:
                st = self.add_spline_tool(
                    nodes,
                    lc=len(self.splinetools),
                    lw=3,
                    can_add_nodes=False,
                )
                self.splinetools.append(st)
            self.status.text(
                f"Loaded {len(self.splinetools)} splines from {self.output_prefix}splines.npy")
            self.render()

        elif evt.keypress == "Ctrl+z":
            if len(self.splinetools) > 0:
                self.splinetools[-1].off()
                self.splinetools.pop()
                self.status.text(f"Removed last spline.")
                self.render()
        
        elif evt.keypress == "Ctrl+c":
            self.remove(self.intermediates)
            self.intermediates.clear()
            self.status.text("Cleared all intermediate shapes.")
            self.render()

        elif evt.keypress == "Ctrl+g":
            all_pts = []
            for st in self.splinetools:
                nodes = st.nodes()
                coords = Spline(nodes, res=self.n_intermediates+1).coordinates
                all_pts.append(coords)
            all_pts = np.array(all_pts)
            self.remove(self.intermediates)
            self.intermediates.clear()
            if len(all_pts) == 0:
                self.status.text("No splines defined. Use Ctrl+a to add a spline.")
                self.render()
                return
            warped0 = self.meshes[0].clone().wireframe().lighting("off").c("k4").alpha(1/n/5)
            for i in range(1, self.n_intermediates+1):
                sources = all_pts[:, i - 1]
                targets = all_pts[:, i]
                warped0.warp(sources, targets, mode=self.mode)
                self.intermediates.append(warped0.clone())
            self.status.text(f"Generated {len(self.intermediates)} intermediate shapes.")
            self.add(self.intermediates).render()


########################################################################################
class Slicer2DPlotter(Plotter):
    """
    A single slice of a Volume which always faces the camera,
    but at the same time can be oriented arbitrarily in space.
    """

    def __init__(self, vol: vedo.Volume, levels=(None, None), histo_color="red4", **kwargs):
        """
        A single slice of a Volume which always faces the camera,
        but at the same time can be oriented arbitrarily in space.

        Arguments:
            vol : (Volume)
                the Volume object to be isosurfaced.
            levels : (list)
                window and color levels
            histo_color : (color)
                histogram color, use `None` to disable it
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        <img src="https://vedo.embl.es/images/volumetric/read_volume3.jpg" width="500">
        """

        if "shape" not in kwargs:
            custom_shape = [  # define here the 2 rendering rectangle spaces
                dict(bottomleft=(0.0, 0.0), topright=(1, 1), bg="k9"),  # the full window
                dict(bottomleft=(0.8, 0.8), topright=(1, 1), bg="k8", bg2="lb"),
            ]
            kwargs["shape"] = custom_shape

        if "interactive" not in kwargs:
            kwargs["interactive"] = True

        super().__init__(**kwargs)

        self.user_mode("image")
        self.add_callback("KeyPress", self.on_key_press)

        orig_volume = vol.clone(deep=False)
        self.volume = vol

        self.volume.actor = vtki.new("ImageSlice")

        self.volume.properties = self.volume.actor.GetProperty()
        self.volume.properties.SetInterpolationTypeToLinear()

        self.volume.mapper = vtki.new("ImageResliceMapper")
        self.volume.mapper.SetInputData(self.volume.dataset)
        self.volume.mapper.SliceFacesCameraOn()
        self.volume.mapper.SliceAtFocalPointOn()
        self.volume.mapper.SetAutoAdjustImageQuality(False)
        self.volume.mapper.BorderOff()

        # no argument will grab the existing cmap in vol (or use build_lut())
        self.lut = None
        self.cmap()

        if levels[0] and levels[1]:
            self.lighting(window=levels[0], level=levels[1])

        self.usage_txt = (
            "H                  :rightarrow Toggle this banner on/off\n"
            "Left click & drag  :rightarrow Modify luminosity and contrast\n"
            "SHIFT-Left click   :rightarrow Slice image obliquely\n"
            "SHIFT-Middle click :rightarrow Slice image perpendicularly\n"
            "SHIFT-R            :rightarrow Fly to closest cartesian view\n"
            "SHIFT-U            :rightarrow Toggle parallel projection"
        )

        self.usage = Text2D(
            self.usage_txt, font="Calco", pos="top-left", s=0.8, bg="yellow", alpha=0.25
        )

        hist = None
        if histo_color is not None:
            data = self.volume.pointdata[0]
            arr = data
            if data.ndim == 1:
                # try to reduce the number of values to histogram
                dims = self.volume.dimensions()
                n = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
                n = min(1_000_000, n)
                arr = np.random.choice(self.volume.pointdata[0], n)
                hist = vedo.pyplot.histogram(
                    arr,
                    bins=12,
                    logscale=True,
                    c=histo_color,
                    ytitle="log_10 (counts)",
                    axes=dict(text_scale=1.9),
                ).clone2d(pos="bottom-left", size=0.4)

        axes = kwargs.pop("axes", 7)
        axe = None
        if axes == 7:
            axe = vedo.addons.RulerAxes(
                orig_volume, xtitle="x - ", ytitle="y - ", ztitle="z - "
            )

        box = orig_volume.box().alpha(0.25)

        volume_axes_inset = vedo.addons.Axes(
            box,
            yzgrid=False,
            xlabel_size=0,
            ylabel_size=0,
            zlabel_size=0,
            tip_size=0.08,
            axes_linewidth=3,
            xline_color="dr",
            yline_color="dg",
            zline_color="db",
            xtitle_color="dr",
            ytitle_color="dg",
            ztitle_color="db",
            xtitle_size=0.1,
            ytitle_size=0.1,
            ztitle_size=0.1,
            title_font="VictorMono",
        )

        self.at(0).add(self.volume, box, axe, self.usage, hist)
        self.at(1).add(orig_volume, volume_axes_inset)
        self.at(0)  # set focus at renderer 0

    ####################################################################
    def on_key_press(self, evt):
        """Handle keyboard events"""
        if evt.keypress == "q":
            self.break_interaction()
        elif evt.keypress.lower() == "h":
            t = self.usage
            if len(t.text()) > 50:
                self.usage.text("Press H to show help")
            else:
                self.usage.text(self.usage_txt)
            self.render()

    def cmap(self, lut=None, fix_scalar_range=False) -> "Slicer2DPlotter":
        """
        Assign a LUT (Look Up Table) to colorize the slice, leave it `None`
        to reuse an existing Volume color map.
        Use "bw" for automatic black and white.
        """
        if lut is None and self.lut:
            self.volume.properties.SetLookupTable(self.lut)
        elif isinstance(lut, vtki.vtkLookupTable):
            self.volume.properties.SetLookupTable(lut)
        elif lut == "bw":
            self.volume.properties.SetLookupTable(None)
        self.volume.properties.SetUseLookupTableScalarRange(fix_scalar_range)
        return self

    def alpha(self, value: float) -> "Slicer2DPlotter":
        """Set opacity to the slice"""
        self.volume.properties.SetOpacity(value)
        return self

    def auto_adjust_quality(self, value=True) -> "Slicer2DPlotter":
        """Automatically reduce the rendering quality for greater speed when interacting"""
        self.volume.mapper.SetAutoAdjustImageQuality(value)
        return self

    def slab(self, thickness=0, mode=0, sample_factor=2) -> "Slicer2DPlotter":
        """
        Make a thick slice (slab).

        Arguments:
            thickness : (float)
                set the slab thickness, for thick slicing
            mode : (int)
                The slab type:
                    0 = min
                    1 = max
                    2 = mean
                    3 = sum
            sample_factor : (float)
                Set the number of slab samples to use as a factor of the number of input slices
                within the slab thickness. The default value is 2, but 1 will increase speed
                with very little loss of quality.
        """
        self.volume.mapper.SetSlabThickness(thickness)
        self.volume.mapper.SetSlabType(mode)
        self.volume.mapper.SetSlabSampleFactor(sample_factor)
        return self

    def face_camera(self, value=True) -> "Slicer2DPlotter":
        """Make the slice always face the camera or not."""
        self.volume.mapper.SetSliceFacesCameraOn(value)
        return self

    def jump_to_nearest_slice(self, value=True) -> "Slicer2DPlotter":
        """
        This causes the slicing to occur at the closest slice to the focal point,
        instead of the default behavior where a new slice is interpolated between
        the original slices.
        Nothing happens if the plane is oblique to the original slices.
        """
        self.volume.mapper.SetJumpToNearestSlice(value)
        return self

    def fill_background(self, value=True) -> "Slicer2DPlotter":
        """
        Instead of rendering only to the image border,
        render out to the viewport boundary with the background color.
        The background color will be the lowest color on the lookup
        table that is being used for the image.
        """
        self.volume.mapper.SetBackground(value)
        return self

    def lighting(self, window, level, ambient=1.0, diffuse=0.0) -> "Slicer2DPlotter":
        """Assign the values for window and color level."""
        self.volume.properties.SetColorWindow(window)
        self.volume.properties.SetColorLevel(level)
        self.volume.properties.SetAmbient(ambient)
        self.volume.properties.SetDiffuse(diffuse)
        return self


########################################################################
class RayCastPlotter(Plotter):
    """
    Generate Volume rendering using ray casting.
    """

    def __init__(self, volume, **kwargs):
        """
        Generate a window for Volume rendering using ray casting.

        Arguments:
            volume : (Volume)
                the Volume object to be isosurfaced.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Returns:
            `vedo.Plotter` object.

        Examples:
            - [app_raycaster.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/app_raycaster.py)

            ![](https://vedo.embl.es/images/advanced/app_raycaster.gif)
        """

        super().__init__(**kwargs)

        self.alphaslider0 = 0.33
        self.alphaslider1 = 0.66
        self.alphaslider2 = 1
        self.color_scalarbar = None

        self.properties = volume.properties

        if volume.dimensions()[2] < 3:
            vedo.logger.error("RayCastPlotter: not enough z slices.")
            raise RuntimeError

        smin, smax = volume.scalar_range()
        x0alpha = smin + (smax - smin) * 0.25
        x1alpha = smin + (smax - smin) * 0.5
        x2alpha = smin + (smax - smin) * 1.0

        ############################## color map slider
        # Create transfer mapping scalar value to color
        cmaps = [
            "rainbow", "rainbow_r",
            "viridis", "viridis_r",
            "bone", "bone_r",
            "hot", "hot_r",
            "plasma", "plasma_r",
            "gist_earth", "gist_earth_r",
            "coolwarm", "coolwarm_r",
            "tab10_r",
        ]
        cols_cmaps = []
        for cm in cmaps:
            cols = color_map(range(0, 21), cm, 0, 20)  # sample 20 colors
            cols_cmaps.append(cols)
        Ncols = len(cmaps)
        csl = "k9"
        if sum(get_color(self.background())) > 1.5:
            csl = "k1"

        def slider_cmap(widget=None, _event=""):
            if widget:
                k = int(widget.value)
                volume.cmap(cmaps[k])
            self.remove(self.color_scalarbar)
            self.color_scalarbar = vedo.addons.ScalarBar(
                volume,
                horizontal=True,
                pos=[(0.8, 0), (0.97, 0.1)],
                font_size=0
            )
            self.add(self.color_scalarbar)

        w1 = self.add_slider(
            slider_cmap,
            0, Ncols - 1,
            value=0,
            show_value=False,
            c=csl,
            pos=[(0.8, 0.05), (0.965, 0.05)],
        )
        w1.representation.SetTitleHeight(0.018)

        ############################## alpha sliders
        # Create transfer mapping scalar value to opacity transfer function
        def setOTF():
            otf = self.properties.GetScalarOpacity()
            otf.RemoveAllPoints()
            otf.AddPoint(smin, 0.0)
            otf.AddPoint(smin + (smax - smin) * 0.1, 0.0)
            otf.AddPoint(x0alpha, self.alphaslider0)
            otf.AddPoint(x1alpha, self.alphaslider1)
            otf.AddPoint(x2alpha, self.alphaslider2)
            slider_cmap()

        setOTF()  ################

        def sliderA0(widget, _event):
            self.alphaslider0 = widget.value
            setOTF()

        self.add_slider(
            sliderA0,
            0, 1,
            value=self.alphaslider0,
            pos=[(0.84, 0.1), (0.84, 0.26)],
            c=csl,
            show_value=0,
        )

        def sliderA1(widget, _event):
            self.alphaslider1 = widget.value
            setOTF()

        self.add_slider(
            sliderA1,
            0, 1,
            value=self.alphaslider1,
            pos=[(0.89, 0.1), (0.89, 0.26)],
            c=csl,
            show_value=0,
        )

        def sliderA2(widget, _event):
            self.alphaslider2 = widget.value
            setOTF()

        w2 = self.add_slider(
            sliderA2,
            0, 1,
            value=self.alphaslider2,
            pos=[(0.96, 0.1), (0.96, 0.26)],
            c=csl,
            show_value=0,
            title="Opacity Levels",
        )
        w2.GetRepresentation().SetTitleHeight(0.015)

        # add a button
        def button_func_mode(_obj, _ename):
            s = volume.mode()
            snew = (s + 1) % 2
            volume.mode(snew)
            bum.switch()

        bum = self.add_button(
            button_func_mode,
            pos=(0.89, 0.31),
            states=["  composite   ", "max projection"],
            c=[ "k3", "k6"],
            bc=["k6", "k3"],  # colors of states
            font="Calco",
            size=18,
            bold=0,
            italic=False,
        )
        bum.frame(color="k6")
        bum.status(volume.mode())

        slider_cmap() ############# init call to create scalarbar

        # add histogram of scalar
        plot = CornerHistogram(
            volume,
            bins=25,
            logscale=1,
            c='k5',
            bg='k5',
            pos=(0.78, 0.065),
            lines=True,
            dots=False,
            nmax=3.1415e06,  # subsample otherwise is too slow
        )

        plot.GetPosition2Coordinate().SetValue(0.197, 0.20, 0)
        plot.GetXAxisActor2D().SetFontFactor(0.7)
        plot.GetProperty().SetOpacity(0.5)
        self.add([plot, volume])


#####################################################################################
class IsosurfaceBrowser(Plotter):
    """
    Generate a Volume isosurfacing controlled by a slider.
    """

    def __init__(
        self,
        volume: vedo.Volume,
        isovalue=None,
        scalar_range=(),
        c=None,
        alpha=1,
        lego=False,
        res=50,
        use_gpu=False,
        precompute=False,
        cmap="hot",
        delayed=False,
        sliderpos=4,
        **kwargs,
    ) -> None:
        """
        Generate a `vedo.Plotter` for Volume isosurfacing using a slider.

        Arguments:
            volume : (Volume)
                the Volume object to be isosurfaced.
            isovalues : (float, list)
                isosurface value(s) to be displayed.
            scalar_range : (list)
                scalar range to be used.
            c : str, (list)
                color(s) of the isosurface(s).
            alpha : (float, list)
                opacity of the isosurface(s).
            lego : (bool)
                if True generate a lego plot instead of a surface.
            res : (int)
                resolution of the isosurface.
            use_gpu : (bool)
                use GPU acceleration.
            precompute : (bool)
                precompute the isosurfaces (so slider browsing will be smoother).
            cmap : (str)
                color map name to be used.
            delayed : (bool)
                delay the slider update on mouse release.
            sliderpos : (int)
                position of the slider.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            - [app_isobrowser.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/app_isobrowser.py)

                ![](https://vedo.embl.es/images/advanced/app_isobrowser.gif)
        """

        super().__init__(**kwargs)

        self.slider = None

        ### GPU ################################
        if use_gpu and hasattr(volume.properties, "GetIsoSurfaceValues"):

            if len(scalar_range) == 2:
                scrange = scalar_range
            else:
                scrange = volume.scalar_range()
            delta = scrange[1] - scrange[0]
            if not delta:
                return

            if isovalue is None:
                isovalue = delta / 3.0 + scrange[0]

            ### isovalue slider callback
            def slider_isovalue(widget, _event):
                value = widget.GetRepresentation().GetValue()
                isovals.SetValue(0, value)

            isovals = volume.properties.GetIsoSurfaceValues()
            isovals.SetValue(0, isovalue)
            self.add(volume.mode(5).alpha(alpha).cmap(c))

            self.slider = self.add_slider(
                slider_isovalue,
                scrange[0] + 0.02 * delta,
                scrange[1] - 0.02 * delta,
                value=isovalue,
                pos=sliderpos,
                title="scalar value",
                show_value=True,
                delayed=delayed,
            )

        ### CPU ################################
        else:

            self._prev_value = 1e30

            scrange = volume.scalar_range()
            delta = scrange[1] - scrange[0]
            if not delta:
                return

            if lego:
                res = int(res / 2)  # because lego is much slower
                slidertitle = ""
            else:
                slidertitle = "scalar value"

            allowed_vals = np.linspace(scrange[0], scrange[1], num=res)

            bacts = {}  # cache the meshes so we dont need to recompute
            if precompute:
                delayed = False  # no need to delay the slider in this case

                for value in allowed_vals:
                    value_name = precision(value, 2)
                    if lego:
                        mesh = volume.legosurface(vmin=value)
                        if mesh.ncells:
                            mesh.cmap(cmap, vmin=scrange[0], vmax=scrange[1], on="cells")
                    else:
                        mesh = volume.isosurface(value).color(c).alpha(alpha)
                    bacts.update({value_name: mesh})  # store it

            ### isovalue slider callback
            def slider_isovalue(widget, _event):

                prevact = self.vol_actors[0]
                if isinstance(widget, float):
                    value = widget
                else:
                    value = widget.GetRepresentation().GetValue()

                # snap to the closest
                idx = (np.abs(allowed_vals - value)).argmin()
                value = allowed_vals[idx]

                if abs(value - self._prev_value) / delta < 0.001:
                    return
                self._prev_value = value

                value_name = precision(value, 2)
                if value_name in bacts:  # reusing the already existing mesh
                    # print('reusing')
                    mesh = bacts[value_name]
                else:  # else generate it
                    # print('generating', value)
                    if lego:
                        mesh = volume.legosurface(vmin=value)
                        if mesh.ncells:
                            mesh.cmap(cmap, vmin=scrange[0], vmax=scrange[1], on="cells")
                    else:
                        mesh = volume.isosurface(value).color(c).alpha(alpha)
                    bacts.update({value_name: mesh})  # store it

                self.remove(prevact).add(mesh)
                self.vol_actors[0] = mesh

            ################################################

            if isovalue is None:
                isovalue = delta / 3.0 + scrange[0]

            self.vol_actors = [None]
            slider_isovalue(isovalue, "")  # init call
            if lego:
                if self.vol_actors[0]:
                    self.vol_actors[0].add_scalarbar(pos=(0.8, 0.12))

            self.slider = self.add_slider(
                slider_isovalue,
                scrange[0] + 0.02 * delta,
                scrange[1] - 0.02 * delta,
                value=isovalue,
                pos=sliderpos,
                title=slidertitle,
                show_value=True,
                delayed=delayed,
            )


##############################################################################
class Browser(Plotter):
    """Browse a series of vedo objects by using a simple slider."""

    def __init__(
        self,
        objects=(),
        sliderpos=((0.50, 0.07), (0.95, 0.07)),
        c=None,  # slider color
        slider_title="",
        font="Calco",  # slider font
        resetcam=False,  # resetcam while using the slider
        **kwargs,
    ):
        """
        Browse a series of vedo objects by using a simple slider.

        The input object can be a list of objects or a list of lists of objects.

        Arguments:
            objects : (list)
                list of objects to be browsed.
            sliderpos : (list)
                position of the slider.
            c : (str)
                color of the slider.
            slider_title : (str)
                title of the slider.
            font : (str)
                font of the slider.
            resetcam : (bool)
                resetcam while using the slider.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            ```python
            from vedo import load, dataurl
            from vedo.applications import Browser
            meshes = load(dataurl+'timecourse1d.npy') # python list of Meshes
            plt = Browser(meshes, bg='k')             # vedo.Plotter
            plt.show(interactive=False, zoom='tight') # show the meshes
            plt.play(dt=50)                           # delay in milliseconds
            plt.close()
            ```

        - [morphomatics_tube.py](https://github.com/marcomusy/vedo/tree/master/examples/other/morphomatics_tube.py)
        """
        kwargs.pop("N", 1)
        kwargs.pop("shape", [])
        kwargs.pop("axes", 1)
        super().__init__(**kwargs)

        if isinstance(objects, str):
            objects = vedo.file_io.load(objects)

        self += objects

        if len(objects)>0 and is_sequence(objects[0]):
            nobs = len(objects[0])
            for ob in objects:
                n = len(ob)
                msg = f"in Browser lists must have the same length but found {n} and {nobs}"
                assert len(ob) == nobs, msg
        else:
            nobs = len(objects)
            if nobs:
                objects = [objects]

        self.slider = None
        self.timer_callback_id = None
        self._oldk = None

        # define the slider func ##########################
        def slider_function(_widget=None, _event=None):

            k = int(self.slider.value)

            if k == self._oldk:
                return  # no change
            self._oldk = k

            n = len(objects)
            m = len(objects[0])
            for i in range(n):
                for j in range(m):
                    ak = objects[i][j]
                    try:
                        if j == k:
                            ak.on()
                            akon = ak
                        else:
                            ak.off()
                    except AttributeError:
                        pass

            try:
                tx = str(k)
                if slider_title:
                    tx = slider_title + " " + tx
                elif n == 1 and akon.filename:
                    tx = akon.filename.split("/")[-1]
                    tx = tx.split("\\")[-1]  # windows os
                elif akon.name:
                    tx = ak.name + " " + tx
            except:
                pass
            self.slider.title = tx

            if resetcam:
                self.reset_camera()
            self.render()

        ##################################################

        self.slider_function = slider_function
        self.slider = self.add_slider(
            slider_function,
            0.5,
            nobs - 0.5,
            pos=sliderpos,
            font=font,
            c=c,
            show_value=False,
        )
        self.slider.GetRepresentation().SetTitleHeight(0.020)
        slider_function()  # init call

    def play(self, dt=100):
        """Start playing the slides at a given speed."""
        self.timer_callback_id = self.add_callback("timer", self.slider_function)
        self.timer_callback("start", dt=dt)
        self.interactive()


#############################################################################################
class FreeHandCutPlotter(Plotter):
    """A tool to edit meshes interactively."""

    # thanks to Jakub Kaminski for the original version of this script
    def __init__(
        self,
        mesh: Union[vedo.Mesh, vedo.Points],
        splined=True,
        font="Bongas",
        alpha=0.9,
        lw=4,
        lc="red5",
        pc="red4",
        c="green3",
        tc="k9",
        tol=0.008,
        **options,
    ):
        """
        A `vedo.Plotter` derived class which edits polygonal meshes interactively.

        Can also be invoked from command line with:

        ```bash
        vedo --edit https://vedo.embl.es/examples/data/porsche.ply
        ```

        Usage:
            - Left-click and hold to rotate
            - Right-click and move to draw line
            - Second right-click to stop drawing
            - Press "c" to clear points
            -       "z/Z" to cut mesh (Z inverts inside-out the selection area)
            -       "L" to keep only the largest connected surface
            -       "s" to save mesh to file (tag `_edited` is appended to filename)
            -       "u" to undo last action
            -       "h" for help, "i" for info

        Arguments:
            mesh : (Mesh, Points)
                The input Mesh or pointcloud.
            splined : (bool)
                join points with a spline or a simple line.
            font : (str)
                Font name for the instructions.
            alpha : (float)
                transparency of the instruction message panel.
            lw : (str)
                selection line width.
            lc : (str)
                selection line color.
            pc : (str)
                selection points color.
            c : (str)
                background color of instructions.
            tc : (str)
                text color of instructions.
            tol : (int)
                tolerance of the point proximity.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            - [cut_freehand.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/cut_freehand.py)

                ![](https://vedo.embl.es/images/basic/cutFreeHand.gif)
        """

        if not isinstance(mesh, Points):
            vedo.logger.error("FreeHandCutPlotter input must be Points or Mesh")
            raise RuntimeError()

        super().__init__(**options)

        self.mesh = mesh
        self.mesh_prev = mesh
        self.splined = splined
        self.linecolor = lc
        self.linewidth = lw
        self.pointcolor = pc
        self.color = c
        self.alpha = alpha

        self.msg = "Right-click and move to draw line\n"
        self.msg += "Second right-click to stop drawing\n"
        self.msg += "Press L to extract largest surface\n"
        self.msg += "        z/Z to cut mesh (s to save)\n"
        self.msg += "        c to clear points, u to undo"
        self.txt2d = Text2D(self.msg, pos="top-left", font=font, s=0.9)
        self.txt2d.c(tc).background(c, alpha).frame()

        self.idkeypress = self.add_callback("KeyPress", self._on_keypress)
        self.idrightclck = self.add_callback("RightButton", self._on_right_click)
        self.idmousemove = self.add_callback("MouseMove", self._on_mouse_move)
        self.drawmode = False
        self.tol = tol  # tolerance of point distance
        self.cpoints = []
        self.points = None
        self.spline = None
        self.jline = None
        self.topline = None
        self.top_pts = []

    def init(self, init_points):
        """Set an initial number of points to define a region"""
        if isinstance(init_points, Points):
            self.cpoints = init_points.coordinates
        else:
            self.cpoints = np.array(init_points)
        self.points = Points(self.cpoints, r=self.linewidth).c(self.pointcolor).pickable(0)
        if self.splined:
            self.spline = Spline(self.cpoints, res=len(self.cpoints) * 4)
        else:
            self.spline = Line(self.cpoints)
        self.spline.lw(self.linewidth).c(self.linecolor).pickable(False)
        self.jline = Line(self.cpoints[0], self.cpoints[-1], lw=1, c=self.linecolor).pickable(0)
        self.add([self.points, self.spline, self.jline]).render()
        return self

    def _on_right_click(self, _evt):
        self.drawmode = not self.drawmode  # toggle mode
        if self.drawmode:
            self.txt2d.background(self.linecolor, self.alpha)
        else:
            self.txt2d.background(self.color, self.alpha)
            if len(self.cpoints) > 2:
                self.remove([self.spline, self.jline])
                if self.splined:  # show the spline closed
                    self.spline = Spline(self.cpoints, closed=True, res=len(self.cpoints) * 4)
                else:
                    self.spline = Line(self.cpoints, closed=True)
                self.spline.lw(self.linewidth).c(self.linecolor).pickable(False)
                self.add(self.spline)
        self.render()

    def _on_mouse_move(self, evt):
        if self.drawmode:
            cpt = self.compute_world_coordinate(evt.picked2d)  # make this 2d-screen point 3d
            if self.cpoints and mag(cpt - self.cpoints[-1]) < self.mesh.diagonal_size() * self.tol:
                return  # new point is too close to the last one. skip
            self.cpoints.append(cpt)
            if len(self.cpoints) > 2:
                self.remove([self.points, self.spline, self.jline, self.topline])
                self.points = Points(self.cpoints, r=self.linewidth).c(self.pointcolor).pickable(0)
                if self.splined:
                    self.spline = Spline(self.cpoints, res=len(self.cpoints) * 4)  # not closed here
                else:
                    self.spline = Line(self.cpoints)

                if evt.actor:
                    self.top_pts.append(evt.picked3d)
                    self.topline = Points(self.top_pts, r=self.linewidth)
                    self.topline.c(self.linecolor).pickable(False)

                self.spline.lw(self.linewidth).c(self.linecolor).pickable(False)
                self.txt2d.background(self.linecolor)
                self.jline = Line(self.cpoints[0], self.cpoints[-1], lw=1, c=self.linecolor).pickable(0)
                self.add([self.points, self.spline, self.jline, self.topline]).render()

    def _on_keypress(self, evt):
        if evt.keypress.lower() == "z" and self.spline:  # Cut mesh with a ribbon-like surface
            inv = False
            if evt.keypress == "Z":
                inv = True
            self.txt2d.background("red8").text("  ... working ...  ")
            self.render()
            self.mesh_prev = self.mesh.clone()
            tol = self.mesh.diagonal_size() / 2  # size of ribbon (not shown)
            pts = self.spline.coordinates
            n = fit_plane(pts, signed=True).normal  # compute normal vector to points
            rb = Ribbon(pts - tol * n, pts + tol * n, closed=True)
            self.mesh.cut_with_mesh(rb, invert=inv)  # CUT
            self.txt2d.text(self.msg)  # put back original message
            if self.drawmode:
                self._on_right_click(evt)  # toggle mode to normal
            else:
                self.txt2d.background(self.color, self.alpha)
            self.remove([self.spline, self.points, self.jline, self.topline]).render()
            self.cpoints, self.points, self.spline = [], None, None
            self.top_pts, self.topline = [], None

        elif evt.keypress == "L":
            self.txt2d.background("red8")
            self.txt2d.text(" ... removing smaller ... \n ... parts of the mesh ... ")
            self.render()
            self.remove(self.mesh)
            self.mesh_prev = self.mesh
            mcut = self.mesh.extract_largest_region()
            mcut.filename = self.mesh.filename  # copy over various properties
            mcut.name = self.mesh.name
            mcut.scalarbar = self.mesh.scalarbar
            mcut.info = self.mesh.info
            self.mesh = mcut                            # discard old mesh by overwriting it
            self.txt2d.text(self.msg).background(self.color)   # put back original message
            self.add(mcut).render()

        elif evt.keypress == "u":  # Undo last action
            if self.drawmode:
                self._on_right_click(evt)  # toggle mode to normal
            else:
                self.txt2d.background(self.color, self.alpha)
            self.remove([self.mesh, self.spline, self.jline, self.points, self.topline])
            self.mesh = self.mesh_prev
            self.cpoints, self.points, self.spline = [], None, None
            self.top_pts, self.topline = [], None
            self.add(self.mesh).render()

        elif evt.keypress in ("c", "Delete"):
            # clear all points
            self.remove([self.spline, self.points, self.jline, self.topline]).render()
            self.cpoints, self.points, self.spline = [], None, None
            self.top_pts, self.topline = [], None

        elif evt.keypress == "r":  # reset camera and axes
            self.remove(self.axes_instances[0])
            self.axes_instances[0] = None
            self.add_global_axes(axtype=1, c=None)
            self.renderer.ResetCamera()
            self.render()

        elif evt.keypress == "s":
            if self.mesh.filename:
                fname = os.path.basename(self.mesh.filename)
                fname, extension = os.path.splitext(fname)
                fname = fname.replace("_edited", "")
                fname = f"{fname}_edited{extension}"
            else:
                fname = "mesh_edited.vtk"
            self.write(fname)

    def write(self, filename="mesh_edited.vtk") -> "FreeHandCutPlotter":
        """Save the resulting mesh to file"""
        self.mesh.write(filename)
        vedo.logger.info(f"mesh saved to file {filename}")
        return self

    def start(self, *args, **kwargs) -> "FreeHandCutPlotter":
        """Start window interaction (with mouse and keyboard)"""
        acts = [self.txt2d, self.mesh, self.points, self.spline, self.jline]
        self.show(acts + list(args), **kwargs)
        return self


########################################################################
class SplinePlotter(Plotter):
    """
    Interactive drawing of splined curves on meshes.
    """

    def __init__(self, obj, init_points=(), closed=False, splined=True, mode="auto", **kwargs):
        """
        Create an interactive application that allows the user to click points and
        retrieve the coordinates of such points and optionally a spline or line
        (open or closed).
        Input object can be a image file name or a 3D mesh.

        Arguments:
            obj : (Mesh, str)
                The input object can be a image file name or a 3D mesh.
            init_points : (list)
                Set an initial number of points to define a region.
            closed : (bool)
                Close the spline or line.
            splined : (bool)
                Join points with a spline or a simple line.
            mode : (str)
                Set the mode of interaction.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.
        """
        super().__init__(**kwargs)

        self.verbose = True
        self.splined = splined
        self.resolution = None  # spline resolution (None = automatic)
        self.closed = closed
        self.lcolor = "yellow4"
        self.lwidth = 3
        self.pcolor = "purple5"
        self.psize = 10

        self.cpoints = list(init_points)
        self.vpoints = None
        self.line = None

        if isinstance(obj, str):
            self.object = vedo.file_io.load(obj)
        else:
            self.object = obj

        self.mode = mode
        if self.mode == "auto":
            if isinstance(self.object, vedo.Image):
                self.mode = "image"
                self.parallel_projection(True)
            else:
                self.mode = "TrackballCamera"

        t = (
            "Click to add a point\n"
            "Right-click to remove it\n"
            "Drag mouse to change contrast\n"
            "Press c to clear points\n"
            "Press q to continue"
        )
        self.instructions = Text2D(t, pos="bottom-left", c="white", bg="green", font="Calco")

        self += [self.object, self.instructions]

        self.callid1 = self.add_callback("KeyPress", self._key_press)
        self.callid2 = self.add_callback("LeftButtonPress", self._on_left_click)
        self.callid3 = self.add_callback("RightButtonPress", self._on_right_click)


    def points(self, newpts=None) -> Union["SplinePlotter", np.ndarray]:
        """Retrieve the 3D coordinates of the clicked points"""
        if newpts is not None:
            self.cpoints = newpts
            self.update()
            return self
        return np.array(self.cpoints)

    def _on_left_click(self, evt):
        if not evt.actor:
            return
        if evt.actor.name == "points":
            # remove clicked point if clicked twice
            pid = self.vpoints.closest_point(evt.picked3d, return_point_id=True)
            self.cpoints.pop(pid)
            self.update()
            return
        p = evt.picked3d
        self.cpoints.append(p)
        self.update()
        if self.verbose:
            vedo.colors.printc("Added point:", precision(p, 4), c="g")

    def _on_right_click(self, evt):
        if evt.actor and len(self.cpoints) > 0:
            self.cpoints.pop()  # pop removes from the list the last pt
            self.update()
            if self.verbose:
                vedo.colors.printc("Deleted last point", c="r")

    def update(self):
        """Update the plot with the new points"""
        self.remove(self.line, self.vpoints)  # remove old points and spline
        self.vpoints = Points(self.cpoints).ps(self.psize).c(self.pcolor)
        self.vpoints.name = "points"
        self.vpoints.pickable(True)  # to allow toggle
        minnr = 1
        if self.splined:
            minnr = 2
        if self.lwidth and len(self.cpoints) > minnr:
            if self.splined:
                try:
                    self.line = Spline(self.cpoints, closed=self.closed, res=self.resolution)
                except ValueError:
                    # if clicking too close splining might fail
                    self.cpoints.pop()
                    return
            else:
                self.line = Line(self.cpoints, closed=self.closed)
            self.line.c(self.lcolor).lw(self.lwidth).pickable(False)
            self.add(self.vpoints, self.line)
        else:
            self.add(self.vpoints)

    def _key_press(self, evt):
        if evt.keypress == "c":
            self.cpoints = []
            self.remove(self.line, self.vpoints).render()
            if self.verbose:
                vedo.colors.printc("==== Cleared all points ====", c="r", invert=True)

    def start(self) -> "SplinePlotter":
        """Start the interaction"""
        self.update()
        self.show(self.object, self.instructions, mode=self.mode)
        return self

########################################################################
class ImageEditor(Plotter):
    """A simple image editor for applying various filters to images."""

    def __init__(self, filename, **kwargs):
        """Initialize the Image Editor with a given image file."""

        if "shape" not in kwargs:
            kwargs["shape"] = (1, 3)
        if "size" not in kwargs:
            kwargs["size"] = (1800, 600)
        if "title" not in kwargs:
            kwargs["title"] = "vedo - Image Editor"

        vedo.settings.enable_default_keyboard_callbacks = False

        super().__init__(**kwargs)

        self.cutfilter = 0.035  # value for the high cutoff frequency in the filter

        self.filename = filename
        self.image0 = vedo.Image(filename)
        self.image2 = self.image0.clone()
        self.image0.pickable(False)

        self.header = Text2D(
            f"File name:\n {filename[-35:]}\nSize: {self.image2.shape}",
            c="w",
            bg="k2",
            alpha=1,
            font="Calco",
            s=1.2,
        )
        self.instructions = Text2D(
            "Press -----------------------\n"
            "Ctrl+e to enhance\n"
            "Ctrl+m to median\n"
            "Ctrl+f to flip horizontally\n"
            "Ctrl+v to flip vertically\n"
            "Ctrl+t to rotate\n"
            "Ctrl+b to binarize\n"
            "Ctrl+i to invert\n"
            "Ctrl+g to smooth\n"
            "Ctrl+h to print info\n"
            "Ctrl+o to increase luminosity\n"
            "Ctrl+l to decrease luminosity\n"
            "Shift+O to increase contrast\n"
            "Shift+L to decrease contrast\n"
            "Ctrl+d to denoise\n"
            "Ctrl+r to reset\n"
            "Ctrl+s to save image\n"
            "Ctrl+c to open color picker\n"
            "q to quit\n"
            "-----------------------------",
            pos=(0.01, 0.8),
            c="w",
            bg="b4",
            alpha=1,
            font="Calco",
            s=0.8,
        )
        self.status = Text2D(
            pos="bottom-center", c="w", bg="r6", alpha=0.8, font="Calco"
        )

        self.pan = vedo.interactor_modes.MousePan(enable_rotate=False)

        self.add_callback("key", self.key_func)

        self.at(0).add(self.header, self.instructions, self.status)
        self.at(1).add(vedo.Text2D("Input",  font="Calco", c="k5", bg="y5"), self.image0)
        self.at(2).add(vedo.Text2D("Output", font="Calco", c="k5", bg="y5"), self.image2)
        self.user_mode(self.pan).show()
        self.interactor.RemoveObservers("CharEvent")

    def key_func(self, evt):
        """Handle key events for image editing."""
        if evt.keypress in ["q", "Ctrl+q", "Ctrl+w"]:
            self.close()
            return

        elif evt.keypress == "Ctrl+e":
            self.image2.enhance()
            self.status.text("Image enhanced")

        elif evt.keypress == "Ctrl+m":
            self.image2.median()
            self.status.text("Image median filtered")

        elif evt.keypress == "Ctrl+f":
            self.image2.mirror()
            self.status.text("Image flipped horizontally")

        elif evt.keypress == "Ctrl+v":
            self.image2.flip()
            self.status.text("Image flipped vertically")

        elif evt.keypress == "Ctrl+t":
            self.image2.rotate(90)
            self.status.text("Image rotated by 90 degrees")

        elif evt.keypress == "Ctrl+b":
            self.image2.binarize()
            self.status.text("Image binarized")

        elif evt.keypress == "Ctrl+i":
            self.image2.invert()
            self.status.text("Image inverted")

        elif evt.keypress == "Ctrl+g":
            self.image2.smooth(sigma=1)
            self.status.text("Image smoothed")

        elif evt.keypress == "Ctrl+h":
            print(self.image2)
            return

        elif evt.keypress == "Ctrl+o":  # change luminosity
            narray = self.image2.tonumpy(raw=True)
            narray[:] = (narray*1.1).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Increased luminosity")
        elif evt.keypress == "Ctrl+l":  # change luminosity
            narray = self.image2.tonumpy(raw=True)
            narray[:] = (narray*0.9).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Decreased luminosity")

        elif evt.keypress == "O":  # change contrast
            narray = self.image2.tonumpy(raw=True)
            m = np.median(narray)
            narray[:] = ((narray - m) * 1.1 + m).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Increased contrast")
        elif evt.keypress == "L":
            narray = self.image2.tonumpy(raw=True)
            m = np.median(narray)
            narray[:] = ((narray - m) * 0.9 + m).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Decreased contrast")

        elif evt.keypress == "Ctrl+d":  # denoise
            self.image2.filterpass(highcutoff=self.cutfilter)
            self.status.text("Denoised image")

        elif evt.keypress == "Ctrl+r":
            self.image2 = self.image0.clone()
            self.status.text("Image reset to original")

        elif evt.keypress == "Ctrl+s":  # save image
            basename = os.path.basename(self.filename)
            sp = basename.split(".")
            name = f"{sp[0]}_edited.{sp[-1]}"
            self.image2.filename = name
            self.image2.write(name)
            self.status.text(f"Image saved as:\n{name}")
            self.render()
            return

        elif evt.keypress == "Ctrl+c":  # open color picker
            self.color_picker(evt.picked2d, verbose=True)
            return

        else:
            self.status.text("")
            self.render()
            return

        self.at(2).remove("Image").add(self.image2).render()

    def start(self):
        """Start the interactive image editor."""
        self.at(1).reset_camera(0.01).interactive()
        return self
    

########################################################################
class Animation(Plotter):
    """
    A `Plotter` derived class that allows to animate simultaneously various objects
    by specifying event times and durations of different visual effects.

    Arguments:
        total_duration : (float)
            expand or shrink the total duration of video to this value
        time_resolution : (float)
            in seconds, save a frame at this rate
        show_progressbar : (bool)
            whether to show a progress bar or not
        video_filename : (str)
            output file name of the video
        video_fps : (int)
            desired value of the nr of frames per second

    .. warning:: this is still very experimental at the moment.
    """

    def __init__(
        self,
        total_duration=None,
        time_resolution=0.02,
        show_progressbar=True,
        video_filename="animation.mp4",
        video_fps=12,
    ):
        super().__init__()
        self.resetcam = True

        self.events = []
        self.time_resolution = time_resolution
        self.total_duration = total_duration
        self.show_progressbar = show_progressbar
        self.video_filename = video_filename
        self.video_fps = video_fps
        self.bookingMode = True
        self._inputvalues = []
        self._performers = []
        self._lastT = None
        self._lastDuration = None
        self._lastActs = None
        self.eps = 0.00001

    def _parse(self, objs, t, duration):
        if t is None:
            if self._lastT:
                t = self._lastT
            else:
                t = 0.0
        if duration is None:
            if self._lastDuration:
                duration = self._lastDuration
            else:
                duration = 0.0
        if objs is None:
            if self._lastActs:
                objs = self._lastActs
            else:
                vedo.logger.error("Need to specify actors!")
                raise RuntimeError

        objs2 = objs

        if is_sequence(objs):
            objs2 = objs
        else:
            objs2 = [objs]

        # quantize time steps and duration
        t = int(t / self.time_resolution + 0.5) * self.time_resolution
        nsteps = int(duration / self.time_resolution + 0.5)
        duration = nsteps * self.time_resolution

        rng = np.linspace(t, t + duration, nsteps + 1)

        self._lastT = t
        self._lastDuration = duration
        self._lastActs = objs2

        for a in objs2:
            if a not in self.objects:
                self.objects.append(a)

        return objs2, t, duration, rng

    def switch_on(self, acts=None, t=None):
        """Switch on the input list of meshes."""
        return self.fade_in(acts, t, 0)

    def switch_off(self, acts=None, t=None):
        """Switch off the input list of meshes."""
        return self.fade_out(acts, t, 0)

    def fade_in(self, acts=None, t=None, duration=None):
        """Gradually switch on the input list of meshes by increasing opacity."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = lin_interpolate(tt, [t, t + duration], [0, 1])
                self.events.append((tt, self.fade_in, acts, alpha))
        else:
            for a in self._performers:
                if hasattr(a, "alpha"):
                    if a.alpha() >= self._inputvalues:
                        continue
                    a.alpha(self._inputvalues)
        return self

    def fade_out(self, acts=None, t=None, duration=None):
        """Gradually switch off the input list of meshes by increasing transparency."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = lin_interpolate(tt, [t, t + duration], [1, 0])
                self.events.append((tt, self.fade_out, acts, alpha))
        else:
            for a in self._performers:
                if a.alpha() <= self._inputvalues:
                    continue
                a.alpha(self._inputvalues)
        return self

    def change_alpha_between(self, alpha1, alpha2, acts=None, t=None, duration=None):
        """Gradually change transparency for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = lin_interpolate(tt, [t, t + duration], [alpha1, alpha2])
                self.events.append((tt, self.fade_out, acts, alpha))
        else:
            for a in self._performers:
                a.alpha(self._inputvalues)
        return self

    def change_color(self, c, acts=None, t=None, duration=None):
        """Gradually change color for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = get_color(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.color()
                    r = lin_interpolate(tt, [t, t + duration], [col1[0], col2[0]])
                    g = lin_interpolate(tt, [t, t + duration], [col1[1], col2[1]])
                    b = lin_interpolate(tt, [t, t + duration], [col1[2], col2[2]])
                    inputvalues.append((r, g, b))
                self.events.append((tt, self.change_color, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.color(self._inputvalues[i])
        return self

    def change_backcolor(self, c, acts=None, t=None, duration=None):
        """Gradually change backface color for the input list of meshes.
        An initial backface color should be set in advance."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = get_color(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    if a.GetBackfaceProperty():
                        col1 = a.backColor()
                        r = lin_interpolate(tt, [t, t + duration], [col1[0], col2[0]])
                        g = lin_interpolate(tt, [t, t + duration], [col1[1], col2[1]])
                        b = lin_interpolate(tt, [t, t + duration], [col1[2], col2[2]])
                        inputvalues.append((r, g, b))
                    else:
                        inputvalues.append(None)
                self.events.append((tt, self.change_backcolor, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.backColor(self._inputvalues[i])
        return self

    def change_to_wireframe(self, acts=None, t=None):
        """Switch representation to wireframe for the input list of meshes at time `t`."""
        if self.bookingMode:
            acts, t, _, _ = self._parse(acts, t, None)
            self.events.append((t, self.change_to_wireframe, acts, True))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self

    def change_to_surface(self, acts=None, t=None):
        """Switch representation to surface for the input list of meshes at time `t`."""
        if self.bookingMode:
            acts, t, _, _ = self._parse(acts, t, None)
            self.events.append((t, self.change_to_surface, acts, False))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self

    def change_line_width(self, lw, acts=None, t=None, duration=None):
        """Gradually change line width of the mesh edges for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    newlw = lin_interpolate(tt, [t, t + duration], [a.lw(), lw])
                    inputvalues.append(newlw)
                self.events.append((tt, self.change_line_width, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.lw(self._inputvalues[i])
        return self

    def change_line_color(self, c, acts=None, t=None, duration=None):
        """Gradually change line color of the mesh edges for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            col2 = get_color(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.linecolor()
                    r = lin_interpolate(tt, [t, t + duration], [col1[0], col2[0]])
                    g = lin_interpolate(tt, [t, t + duration], [col1[1], col2[1]])
                    b = lin_interpolate(tt, [t, t + duration], [col1[2], col2[2]])
                    inputvalues.append((r, g, b))
                self.events.append((tt, self.change_line_color, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.linecolor(self._inputvalues[i])
        return self

    def change_lighting(self, style, acts=None, t=None, duration=None):
        """Gradually change the lighting style for the input list of meshes.

        Allowed styles are: [metallic, plastic, shiny, glossy, default].
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, c]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                vedo.logger.error(f"Unknown lighting style {style}")

            for tt in rng:
                inputvalues = []
                for a in acts:
                    pr = a.properties
                    aa = pr.GetAmbient()
                    ad = pr.GetDiffuse()
                    asp = pr.GetSpecular()
                    aspp = pr.GetSpecularPower()
                    naa = lin_interpolate(tt, [t, t + duration], [aa, pars[0]])
                    nad = lin_interpolate(tt, [t, t + duration], [ad, pars[1]])
                    nasp = lin_interpolate(tt, [t, t + duration], [asp, pars[2]])
                    naspp = lin_interpolate(tt, [t, t + duration], [aspp, pars[3]])
                    inputvalues.append((naa, nad, nasp, naspp))
                self.events.append((tt, self.change_lighting, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                pr = a.properties
                vals = self._inputvalues[i]
                pr.SetAmbient(vals[0])
                pr.SetDiffuse(vals[1])
                pr.SetSpecular(vals[2])
                pr.SetSpecularPower(vals[3])
        return self

    def move(self, act=None, pt=(0, 0, 0), t=None, duration=None, style="linear"):
        """Smoothly change the position of a specific object to a new point in space."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                vedo.logger.error("in move(), can move only one object.")
            cpos = acts[0].pos()
            pt = np.array(pt)
            dv = (pt - cpos) / len(rng)
            for j, tt in enumerate(rng):
                i = j + 1
                if "quad" in style:
                    x = i / len(rng)
                    y = x * x
                    self.events.append((tt, self.move, acts, cpos + dv * i * y))
                else:
                    self.events.append((tt, self.move, acts, cpos + dv * i))
        else:
            self._performers[0].pos(self._inputvalues)
        return self

    def rotate(self, act=None, axis=(1, 0, 0), angle=0, t=None, duration=None):
        """Smoothly rotate a specific object by a specified angle and axis."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                vedo.logger.error("in rotate(), can move only one object.")
            for tt in rng:
                ang = angle / len(rng)
                self.events.append((tt, self.rotate, acts, (axis, ang)))
        else:
            ax = self._inputvalues[0]
            if ax == "x":
                self._performers[0].rotate_x(self._inputvalues[1])
            elif ax == "y":
                self._performers[0].rotate_y(self._inputvalues[1])
            elif ax == "z":
                self._performers[0].rotate_z(self._inputvalues[1])
        return self

    def scale(self, acts=None, factor=1, t=None, duration=None):
        """Smoothly scale a specific object to a specified scale factor."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                fac = lin_interpolate(tt, [t, t + duration], [1, factor])
                self.events.append((tt, self.scale, acts, fac))
        else:
            for a in self._performers:
                a.scale(self._inputvalues)
        return self

    def mesh_erode(self, act=None, corner=6, t=None, duration=None):
        """Erode a mesh by removing cells that are close to one of the 8 corners
        of the bounding box.
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                vedo.logger.error("in meshErode(), can erode only one object.")
            diag = acts[0].diagonal_size()
            x0, x1, y0, y1, z0, z1 = acts[0].GetBounds()
            corners = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            pcl = acts[0].closest_point(corners[corner])
            dmin = np.linalg.norm(pcl - corners[corner])
            for tt in rng:
                d = lin_interpolate(tt, [t, t + duration], [dmin, diag * 1.01])
                if d > 0:
                    ids = acts[0].closest_point(corners[corner], radius=d, return_point_id=True)
                    if len(ids) <= acts[0].npoints:
                        self.events.append((tt, self.mesh_erode, acts, ids))
        return self

    def play(self):
        """Play the internal list of events and save a video."""

        self.events = sorted(self.events, key=lambda x: x[0])
        self.bookingMode = False

        if self.show_progressbar:
            pb = vedo.ProgressBar(0, len(self.events), c="g")

        if self.total_duration is None:
            self.total_duration = self.events[-1][0] - self.events[0][0]

        if self.video_filename:
            vd = vedo.Video(self.video_filename, fps=self.video_fps, duration=self.total_duration)

        ttlast = 0
        for e in self.events:

            tt, action, self._performers, self._inputvalues = e
            action(0, 0)

            dt = tt - ttlast
            if dt > self.eps:
                self.show(interactive=False, resetcam=self.resetcam)
                if self.video_filename:
                    vd.add_frame()

                if dt > self.time_resolution + self.eps:
                    if self.video_filename:
                        vd.pause(dt)

            ttlast = tt

            if self.show_progressbar:
                pb.print("t=" + str(int(tt * 100) / 100) + "s,  " + action.__name__)

        self.show(interactive=False, resetcam=self.resetcam)
        if self.video_filename:
            vd.add_frame()
            vd.close()

        self.show(interactive=True, resetcam=self.resetcam)
        self.bookingMode = True


########################################################################
class AnimationPlayer(vedo.Plotter):
    """
    A Plotter with play/pause, step forward/backward and slider functionalties.
    Useful for inspecting time series.

    The user has the responsibility to update all actors in the callback function.

    Arguments:
        func :  (Callable)
            a function that passes an integer as input and updates the scene
        irange : (tuple)
            the range of the integer input representing the time series index
        dt : (float)
            the time interval between two calls to `func` in milliseconds
        loop : (bool)
            whether to loop the animation
        c : (list, str)
            the color of the play/pause button
        bc : (list)
            the background color of the play/pause button and the slider
        button_size : (int)
            the size of the play/pause buttons
        button_pos : (float, float)
            the position of the play/pause buttons as a fraction of the window size
        button_gap : (float)
            the gap between the buttons
        slider_length : (float)
            the length of the slider as a fraction of the window size
        slider_pos : (float, float)
            the position of the slider as a fraction of the window size
        kwargs: (dict)
            keyword arguments to be passed to `Plotter`

    Examples:
        - [aspring2_player.py](https://vedo.embl.es/images/simulations/spring_player.gif)
    """

    # Original class contributed by @mikaeltulldahl (Mikael Tulldahl)

    PLAY_SYMBOL        = "    \u23F5   "
    PAUSE_SYMBOL       = "   \u23F8   "
    ONE_BACK_SYMBOL    = " \u29CF"
    ONE_FORWARD_SYMBOL = "\u29D0 "

    def __init__(
        self,
        func,
        irange: tuple,
        dt: float = 1.0,
        loop: bool = True,
        c=("white", "white"),
        bc=("green3", "red4"),
        button_size=25,
        button_pos=(0.5, 0.04),
        button_gap=0.055,
        slider_length=0.5,
        slider_pos=(0.5, 0.055),
        **kwargs,
    ):
        super().__init__(**kwargs)

        min_value, max_value = np.array(irange).astype(int)
        button_pos = np.array(button_pos)
        slider_pos = np.array(slider_pos)

        self._func = func

        self.value = min_value - 1
        self.min_value = min_value
        self.max_value = max_value
        self.dt = max(dt, 1)
        self.is_playing = False
        self._loop = loop

        self.timer_callback_id = self.add_callback(
            "timer", self._handle_timer, enable_picking=False
        )
        self.timer_id = None

        self.play_pause_button = self.add_button(
            self.toggle,
            pos=button_pos,  # x,y fraction from bottom left corner
            states=[self.PLAY_SYMBOL, self.PAUSE_SYMBOL],
            font="Kanopus",
            size=button_size,
            bc=bc,
        )
        self.button_oneback = self.add_button(
            self.onebackward,
            pos=(-button_gap, 0) + button_pos,
            states=[self.ONE_BACK_SYMBOL],
            font="Kanopus",
            size=button_size,
            c=c,
            bc=bc,
        )
        self.button_oneforward = self.add_button(
            self.oneforward,
            pos=(button_gap, 0) + button_pos,
            states=[self.ONE_FORWARD_SYMBOL],
            font="Kanopus",
            size=button_size,
            bc=bc,
        )
        d = (1 - slider_length) / 2
        self.slider: SliderWidget = self.add_slider(
            self._slider_callback,
            self.min_value,
            self.max_value - 1,
            value=self.min_value,
            pos=[(d - 0.5, 0) + slider_pos, (0.5 - d, 0) + slider_pos],
            show_value=False,
            c=bc[0],
            alpha=1,
        )

    def pause(self) -> None:
        """Pause the animation."""
        self.is_playing = False
        if self.timer_id is not None:
            self.timer_callback("destroy", self.timer_id)
            self.timer_id = None
        self.play_pause_button.status(self.PLAY_SYMBOL)

    def resume(self) -> None:
        """Resume the animation."""
        if self.timer_id is not None:
            self.timer_callback("destroy", self.timer_id)
        self.timer_id = self.timer_callback("create", dt=int(self.dt))
        self.is_playing = True
        self.play_pause_button.status(self.PAUSE_SYMBOL)

    def toggle(self, _obj, _evt) -> None:
        """Toggle between play and pause."""
        if not self.is_playing:
            self.resume()
        else:
            self.pause()

    def oneforward(self, _obj, _evt) -> None:
        """Advance the animation by one frame."""
        self.pause()
        self.set_frame(self.value + 1)

    def onebackward(self, _obj, _evt) -> None:
        """Go back one frame in the animation."""
        self.pause()
        self.set_frame(self.value - 1)

    def set_frame(self, value: int) -> None:
        """Set the current value of the animation."""
        if self._loop:
            if value < self.min_value:
                value = self.max_value - 1
            elif value >= self.max_value:
                value = self.min_value
        else:
            if value < self.min_value:
                self.pause()
                value = self.min_value
            elif value >= self.max_value - 1:
                value = self.max_value - 1
                self.pause()

        if self.value != value:
            self.value = value
            self.slider.value = value
            self._func(value)

    def _slider_callback(self, widget: SliderWidget, _: str) -> None:
        self.pause()
        self.set_frame(int(round(widget.value)))

    def _handle_timer(self, _evt=None) -> None:
        self.set_frame(self.value + 1)

    def stop(self) -> "AnimationPlayer":
        """
        Stop the animation timers, remove buttons and slider.
        Behave like a normal `Plotter` after this.
        """
        # stop timer
        if self.timer_id is not None:
            self.timer_callback("destroy", self.timer_id)
            self.timer_id = None

        # remove callbacks
        self.remove_callback(self.timer_callback_id)

        # remove buttons
        self.slider.off()
        self.renderer.RemoveActor(self.play_pause_button.actor)
        self.renderer.RemoveActor(self.button_oneback.actor)
        self.renderer.RemoveActor(self.button_oneforward.actor)
        return self


########################################################################
class Clock(vedo.Assembly):
    """Create a clock with current time or user provided time."""

    def __init__(self, h=None, m=None, s=None, font="Quikhand", title="", c="k"):
        """
        Create a clock with current time or user provided time.

        Arguments:
            h : (int)
                hours in range [0,23]
            m : (int)
                minutes in range [0,59]
            s : (int)
                seconds in range [0,59]
            font : (str)
                font type
            title : (str)
                some extra text to show on the clock
            c : (str)
                color of the numbers

        Example:
            ```python
            import time
            from vedo import show
            from vedo.applications import Clock
            clock = Clock()
            plt = show(clock, interactive=False)
            for i in range(10):
                time.sleep(1)
                clock.update()
                plt.render()
            plt.close()
            ```
            ![](https://vedo.embl.es/images/feats/clock.png)
        """
        import time

        self.elapsed = 0
        self._start = time.time()

        wd = ""
        if h is None and m is None:
            t = time.localtime()
            h = t.tm_hour
            m = t.tm_min
            s = t.tm_sec
            if not title:
                d = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                wd = f"{d[t.tm_wday]} {t.tm_mday}/{t.tm_mon}/{t.tm_year} "

        h = int(h) % 24
        m = int(m) % 60
        t = (h * 60 + m) / 12 / 60

        alpha = 2 * np.pi * t + np.pi / 2
        beta = 12 * 2 * np.pi * t + np.pi / 2

        x1, y1 = np.cos(alpha), np.sin(alpha)
        x2, y2 = np.cos(beta), np.sin(beta)
        if s is not None:
            s = int(s) % 60
            gamma = s * 2 * np.pi / 60 + np.pi / 2
            x3, y3 = np.cos(gamma), np.sin(gamma)

        ore = Line([0, 0], [x1, y1], lw=14, c="red4").scale(0.5).mirror()
        minu = Line([0, 0], [x2, y2], lw=7, c="blue3").scale(0.75).mirror()
        secs = None
        if s is not None:
            secs = Line([0, 0], [x3, y3], lw=1, c="k").scale(0.95).mirror()
            secs.z(0.003)
        back1 = vedo.shapes.Circle(res=180, c="k5")
        back2 = vedo.shapes.Circle(res=12).mirror().scale(0.84).rotate_z(-360 / 12)
        labels = back2.labels(range(1, 13), justify="center", font=font, c=c, scale=0.14)
        txt = vedo.shapes.Text3D(wd + title, font="VictorMono", justify="top-center", s=0.07, c=c)
        txt.pos(0, -0.25, 0.001)
        labels.z(0.001)
        minu.z(0.002)
        super().__init__([back1, labels, ore, minu, secs, txt])
        self.name = "Clock"

    def update(self, h=None, m=None, s=None) -> "Clock":
        """Update clock with current or user time."""
        import time
        parts = self.unpack()
        self.elapsed = time.time() - self._start

        if h is None and m is None:
            t = time.localtime()
            h = t.tm_hour
            m = t.tm_min
            s = t.tm_sec

        h = int(h) % 24
        m = int(m) % 60
        t = (h * 60 + m) / 12 / 60

        alpha = 2 * np.pi * t + np.pi / 2
        beta = 12 * 2 * np.pi * t + np.pi / 2

        x1, y1 = np.cos(alpha), np.sin(alpha)
        x2, y2 = np.cos(beta), np.sin(beta)
        if s is not None:
            s = int(s) % 60
            gamma = s * 2 * np.pi / 60 + np.pi / 2
            x3, y3 = np.cos(gamma), np.sin(gamma)

        pts2 = parts[2].coordinates
        pts2[1] = [-x1 * 0.5, y1 * 0.5, 0.001]
        parts[2].coordinates = pts2

        pts3 = parts[3].coordinates
        pts3[1] = [-x2 * 0.75, y2 * 0.75, 0.002]
        parts[3].coordinates = pts3

        if s is not None:
            pts4 = parts[4].coordinates
            pts4[1] = [-x3 * 0.95, y3 * 0.95, 0.003]
            parts[4].coordinates = pts4

        return self
