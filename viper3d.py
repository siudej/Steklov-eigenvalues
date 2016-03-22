""" Custom plotter built on top of viper. """

import vtk
from viper import Viper
import vtk.util.numpy_support as VN
import glob
import re


class Viper3D (Viper):

    def __init__(self, mesh, val, **kwargs):
        self.initcommon(kwargs)
        self.mesh = mesh
        self.x = val
        self.vtkgrid = self.make_vtk_grid(self.mesh)
        self.vtkgrid.GetPointData().SetScalars(VN.numpy_to_vtk(self.x))
        self.vmin = self.x.min()
        self.vmax = self.x.max()
        self.lut = self._make_lut(autorange=(self.vmin, self.vmax))
        self._update = self.update
        self.scalarbar = self._make_scalarbar(self.lut)
        self.scalarbar.VisibilityOn()

    def update(x): return x

    def Contours(self, num, opacity=0.2):
        contour = vtk.vtkMarchingContourFilter()
        contour.SetInput(self.vtkgrid)
        r = max(abs(self.vmin), abs(self.vmax))
        if num % 2 == 0 or self.vmin * self.vmax >= 0:
            contour.GenerateValues(
                num, (self.vmin + r / num, self.vmax - r / num))
        elif num == 1:
            contour.SetValue(0, 0)
        else:
            r = r - r / num
            contour.GenerateValues(num, -r, r)
        contour.ComputeScalarsOn()
        contour.UseScalarTreeOn()
        contour.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInput(contour.GetOutput())
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(normals.GetOutput())
        mapper.SetLookupTable(self.lut)
        mapper.SetScalarRange(self.vmin, self.vmax)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(3)
        return actor

    def Domain(self, opacity=1, mesh=False):
        domain = vtk.vtkGeometryFilter()
        domain.SetInput(self.vtkgrid)
        domain.Update()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInput(domain.GetOutput())
        # edges
        edges = vtk.vtkFeatureEdges()
        edges.SetInput(normals.GetOutput())
        edges.ManifoldEdgesOff()
        if mesh:
            edges.ManifoldEdgesOn()
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOff()
        edges.FeatureEdgesOff()
        edges.SetFeatureAngle(1)
        # mapper for domain
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(normals.GetOutput())
        mapper.SetLookupTable(self.lut)
        mapper.SetScalarRange(self.vmin, self.vmax)
        # actor for domain
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        # mapper for edges
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(edges.GetOutput())
        if self.mesh.topology().dim() == 3:
            mapper.ScalarVisibilityOff()
        else:
            mapper.SetLookupTable(self.lut)
            mapper.SetScalarRange(self.vmin, self.vmax)
        # actor for domain
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper)
        actor2.GetProperty().SetOpacity((1 + opacity) / 2)
        if mesh:
            actor2.GetProperty().SetOpacity(opacity)
        return [actor, actor2]

    def Render(self, actors):
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        for a in actors:
            self.ren.AddActor(a)
        # self.ren.AddActor2D(self.scalarbar)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(600, 600)
        self.renWin.SetWindowName(self.title)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.AddObserver("KeyPressEvent", self.key_press_methods)
        self.iren.Initialize()
        self.renWin.Render()
        self.renWin.SetWindowName(self.title)

    def write_png(self, filename=None):
        large = vtk.vtkRenderLargeImage()
        large.SetInput(self.ren)
        large.SetMagnification(3)
        png = vtk.vtkPNGWriter()
        count = 0
        if filename is None:
            files = glob.glob('image*.png')
            if len(files) > 0:
                numbers = [int
                           (re.sub(r'image(\d+)\.png', r'\1', f))
                           for f in files]
                count = max(numbers) + 1
            filename = 'image{:0>4}.png'.format(count)
        png.SetFileName(filename)
        png.SetInputConnection(large.GetOutputPort())
        png.Write()

    def ContourPlot(self, num, mesh=False, edges=True):
        if self.mesh.topology().dim() == 3:
            opacity = 0.2
        else:
            opacity = 1
            mesh = False
        actor = self.Contours(num, opacity)
        actors = self.Domain(0.3, mesh)
        if not edges:
            actors.pop(1)
        actors.append(actor)
        self.Render(actors)

    def SimplePlot(self, num):
        actor = self.Contours(num, 1)
        self.Render([actor])

    def SurfPlot(self):
        if self.mesh.topology().dim() == 2:
            self.vtkgrid = self.warp_scalar(self.x, self.vmax - self.vmin)
        actor = self.Domain()[0]
        self.Render([actor])

    def Plot(self, opacity=1):
        actor = self.Domain(opacity=opacity)[0]
        self.Render([actor])

    def MeshPlot(self, mesh=True):
        actor = self.Domain(mesh=mesh)
        self.Render(actor)
