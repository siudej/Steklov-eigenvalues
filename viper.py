#!/usr/bin/env python

# Copyright (C) 2006-2011 Ola Skavhaug and Simula Research Laboratory
#
# This file is part of Viper.
#
# Viper is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Viper is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Viper. If not, see <http://www.gnu.org/licenses/>.

__cite__      = """Ola Skavhaug, Viper Visualization Software,
http://www.fenics.org/wiki/viper/"""
__version__ = "1.0.1"

r"""
Viper

A simple mesh plotter and run--time visualization module for plotting and
saving simulation data. The class C{Viper} can visualize solutions given as
numpy arrays, and meshes that provide the two methods C{cells()} and
C{coordinates()}. These methods should return numpy arrays specifying the
node-element ordering and coordinates of the nodes, respectively.

Citation: %s

""" % __cite__

"""TODO: I need to generalize vtkgrid and darr such that multiples of these can
be visualized at once. The fix is probably to construct some pipes (as Python
lists?):
    Source: the input (like a numpy array or vtkvoid array
    Filter chain: The filters applied to the source
"""

class SimpleMesh(object):
    """Simple mesh class used when Viper called with coordinate and cell arrays
    only."""
    def __init__(self, c, e):
        assert isinstance(c, numpy.ndarray)
        assert isinstance(e, numpy.ndarray)
        assert c.dtype=='d'
        assert e.dtype=='i'
        assert c.shape[1] == 3
        assert e.shape[1] in (2,3,4)
        self.c = c
        self.e = e

    def cells(self): return self.e # Return element array

    def coordinates(self): return self.c # Return coordiantes array

import vtk, numpy, os, math
#from vtk.util.colors import tomato

_viper = None

#lutdir = os.path.join(os.path.split(__file__)[0], "data")
lutdir = os.path.join(os.path.split(__file__)[0], "")

class Viper(object):
    """Simple real-time vtk plotter."""

    def __init__(self, *args, **kwargs):
        """
Create plotter.
__init__(mesh=mymesh):
    plot mymesh
__init__(x=x, mesh=mymesh):
    plot scalar of vector field x over mymesh
__init__(x=x, mesh=mymesh, displacement=d):
    plot scalar of vector field x over mymesh and use d as a displacement vector field
__init__(coordinates=coords, cells=elements):
    plot mesh given in the two numpy arrays coords and elements
__init__(..., cutplane_origin = origin, cutplane_normal = normal)
    plot a cut plane of a 3D scalar plot. Set the origin and normal to define the cut plane.
    The normal defaults to (0,0,1)

Additional parameters: vmin, vmax, lutfile, wireframe, celltype, title
        """
        if len(args) > 0:
            nargs = len(args)
            if not nargs > 0:
                raise RuntimeError, "No arguments to Viper, giving up"
            kwargs["mesh"] = args[0]
            if nargs > 1:
                kwargs["x"] = args[1]
            if nargs > 2:
                kwargs["vmin"] = args[2]
            if nargs > 3:
                kwargs["vmax"] = args[3]

        mesh = kwargs.get("mesh", None)
        x = kwargs.get("x", None)
        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
        wireframe = kwargs.get("wireframe", False)
        self.warpscalar = kwargs.get("warpscalar", False)
        displacement = kwargs.get("displacement", None)
        celltype = kwargs.get("celltype", None)
        coordinates = kwargs.get("coordinates", None)
        cells = kwargs.get("cells", None)
        sf = kwargs.get("scalefactor", 2.0)
        self.vtkgrid = None
        self.filter = None

        self._update = self.update
        self.initcommon(kwargs)
        self._x_shape = ()

        if mesh is None:
            if coordinates is not None and cells is not None:
                mesh = SimpleMesh(coordinates, cells)

        self.mesh = mesh
        if x is None:
            x = numpy.zeros(len(mesh.coordinates()), dtype='d')
            wireframe=True
        self.x = x
        if self.mesh is not None:
            self.vtkgrid = self.make_vtk_grid(self.mesh, cell_type=celltype)
            self.filter = self.vtkgrid

        vmin = kwargs.get("vmin", x.min())
        vmax = kwargs.get("vmax", x.max())
        minmax = vmax - vmin

        self.displacement = self._resolve_displacement(displacement)

        self.frame = kwargs.get("frame", None)

        # Extract cut plane information
        cutplane_origin = kwargs.get("cutplane_origin", None)
        cutplane_normal = kwargs.get("cutplane_normal", (0,0,1))

        if len(x.shape) == 1:
            if self.warpscalar and minmax > 0:
                self.filter = self.warp_scalar(self.x, minmax)

            # If plot a cut plane
            if not cutplane_origin is None:
                self.filter = self._cutplane_filter(cutplane_origin, cutplane_normal)
            (self.iren, self.renWin, self.ren) = self.simple_plotter(self.filter,
                                                                     vmin, vmax,
                                                                     wireframe)
            if not cutplane_origin is None:
                # Position the camera normal to the plane with focus on the center of the cutplane
                cutplane_normal    = numpy.asarray(cutplane_normal,dtype = 'd')
                camera_focal_point = numpy.asarray(self.filter.GetCenter(), dtype = 'd')

                #Normalize the cutplane normal
                cutplane_normal /= numpy.sqrt(numpy.dot(cutplane_normal,cutplane_normal))

                # Check which component is the largest in the cutplane normal
                if cutplane_normal.argmax() in [0,1]:
                    # If the largest direction is in the x or y direction, let z be the up direction
                    up_direction = numpy.array([0,0,1])
                else:
                    # If the largest direction is in the z direction, let y be the up direction
                    up_direction = numpy.array([0,1,0])

                # Set view_up as a projection of the cutplane normal onto the up_direction
                # NOTE: Actually no need for the projection, vtk do the projection for you...
                view_up = up_direction - numpy.dot(up_direction,cutplane_normal)*cutplane_normal

                # Set the camera position
                self.ren.GetActiveCamera().SetPosition(camera_focal_point + cutplane_normal)
                self.ren.GetActiveCamera().SetFocalPoint(camera_focal_point)
                self.ren.GetActiveCamera().SetViewUp(view_up)
                self.ren.ResetCamera(self.filter.GetBounds())
                # It is sometimes nice to dolly the view but sometimes not
                #self.ren.GetActiveCamera().Dolly(1.5)

        else:
            self.mode="vector"
            if self.mesh is not None:
                coords = self.mesh.coordinates()
            else:
                coords = coordinates
            (self.iren, self.renWin, self.ren) = self.vector_plotter(coords, x, vmin, vmax, wireframe, sf=sf)

        self.ren.GlobalWarningDisplayOff()
        self._update(x)
        if self.frame is None:
            self.iren.Initialize()

    def init_from_file(self, filename):
        """Construct a scalar field over a unstructured grid from vtk file."""
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        self.vtkgrid = reader.GetOutput()
        self.filter = self.vtkgrid
        pd = self.vtkgrid.GetPointData()
        scalars = pd.GetScalars()
        n = scalars.GetNumberOfTuples()
        x = numpy.zeros(n)
        scalars.ExportToVoidPointer(x)
        self.x = x
        self.displacement = None

        self.lutfile = "gauss_120.lut"
        self.refs = []
        self.basename = "plot"

        self.elevator = 0
        self.elevator_sign = 1
        self.set_camera_movement()
        self.darr = vtk.vtkDoubleArray()

        (self.iren, self.renWin, self.ren) = self.simple_plotter(self.vtkgrid, min(x), max(x), False)
        self._update(self.x)

    def clear(self):
        """Remove all plot objects."""
        self.clear_spheres()
        self.clear_polygons()
        self.clear_scalars()
        self.clear_vectors()

    def reset_camera(self):
        self.ren.ResetCamera()

    def clear_spheres(self):
        """Remove all spheres."""
        self.sphere_data.RemoveAllInputs()
        for actor in self.sphere_actors:
            self.ren.RemoveActor(actor)
        self.sphere_actors = [ vtk.vtkActor() ]
        self.ren.AddActor(self.sphere_actors[-1])
        self.update()

    def clear_polygons(self):
        """Remove all polygons."""
        self.polygon_data.RemoveAllInputs()
        for actor in self.polygon_actors:
            self.ren.RemoveActor(actor)
        self.polygon_actors = [ vtk.vtkActor() ]
        self.update()

    def clear_scalars(self):
        """Remove scalars."""
        for actor in self.scalar_actors:
            self.ren.RemoveActor(actor)
        self.scalar_actors = [ vtk.vtkActor() ]
        self.update()

    def clear_vectors(self):
        """Remove scalars."""
        for actor in self.vector_actors:
            self.ren.RemoveActor(actor)
        self.vector_actors = [vtk.vtkActor()]
        self.update()

    def set_vector(self, mesh, x):
        self.clear()
        self.vector_plotter(mesh.coordinates(), x, x.min(), x.max())
        self.update()

    def set_scalar(self, mesh, x):
        self.clear()
        self.vtkgrid = self.make_vtk_grid(mesh)
        self.simple_plotter(self.vtkgrid, x.min(), x.max())
        self.update(x)

    def add_scalar(self, mesh, x):
        vtkgrid = self.make_vtk_grid(mesh)
        idx = len(self.scalar_actors)
        self.scalar_actors.append(vtk.vtkActor())
        self.update_scalar_mapper(vtkgrid, idx)
        self.ren.AddActor(self.scalar_actors[idx])

    def set_mesh(self, mesh):
        self.clear()
        self.vtkgrid = self.make_vtk_grid(mesh)
        self.simple_plotter(self.vtkgrid, 0,0, True)
        x = numpy.zeros(len(mesh.coordinates()))
        self.update(x)

    def _resolve_displacement(self, d):
        if d is None:
            return None

        if len(d.shape) > 1 and d.shape[1] > 1:
            d = self.vec3d(d)
            self.filter = self.warp_vector(d)
            return d
        else:
            self.filter = self.warp_scalar(d)
            return d

    def initcommon(self, kwargs):
        self.displacement = kwargs.get("displacement", None)
        self.title = kwargs.get("title", "FEniCS Viper")
        self.rescale = kwargs.get("rescale", False)
        self.lutfile = kwargs.get("lutfile", "gauss_120.lut")
        self.frame = kwargs.get("frame", None)
        self.axes_on = kwargs.get("axes", False)
        self.window_size = kwargs.get("size", (600, 400))
        self.args = kwargs.copy()
        self.filters = []
        self.refs = []
        self.basename = kwargs.get("basename", "plot")
        self.is_writer = False
        self.outline = False
        self.vertex_plot = True
        self.plottype = None
        self.mode = None
        self.iren = None
        self.ren = None
        self.renWin = None
        #self.rescale = False

        # paramaters to control camera movement, default is no movement
        self.elevator = 0
        self.elevator_sign = 1
        self.set_camera_movement()
        self.darr = vtk.vtkDoubleArray()

        self.scalar_actors = [ vtk.vtkActor() ]
        self.vector_actors = [ vtk.vtkActor() ]
        self.sphere_actors = [ vtk.vtkActor() ]
        self.polygon_actors = [ vtk.vtkActor() ]


    def warp_scalar(self, d, minmax=1.0):
        if len(d.shape) > 1 and d.shape[1] > 1:
            raise ValueError, "Wrong shape in scalar displacement field"
        self.disp_arr = vtk.vtkDoubleArray()
        self.disp_arr.SetNumberOfComponents(1)
        self.disp_arr.SetVoidArray(d, d.shape[0], 1)
        self.vtkgrid.GetPointData().SetScalars(self.disp_arr)
        if not hasattr(self, "warp"):
            self.warp = vtk.vtkWarpScalar()
            self.warp.SetInput(self.vtkgrid)
            self.warpdata = self.warp.GetOutput()
        self.warp.SetScaleFactor(1.0/minmax)
        return self.warpdata

    def warp_vector(self, d):
        if not d.shape == (len(self.x), 3):
            raise ValueError, "Wrong shape in vector displacement field (must be 3d)"

        self.disp_arr = vtk.vtkDoubleArray()
        self.disp_arr.SetNumberOfComponents(3)
        self.disp_arr.SetVoidArray(d, d.shape[0]*d.shape[1], 1)
        self.vtkgrid.GetPointData().SetVectors(self.disp_arr)

        self.warp = vtk.vtkWarpVector()
        self.warp.SetInput(self.vtkgrid)
        self.warp.SetScaleFactor(1.0)
        self.warpdata = self.warp.GetOutput()
        return self.warpdata

    def set_contour(self, nlevels=5):
        self.contour = vtk.vtkContourFilter()

    def init_writer(self, filebasename="simulation"):
        """Initialize the simple vtk file writer for storing unstructured grids and fields to
        file."""
        self.is_writer = True

        writepath = os.path.dirname(filebasename)
        if writepath and not os.path.isdir(writepath):
            os.makedirs(writepath)

        self.writer = vtk.vtkUnstructuredGridWriter()
        self.writer.SetFileTypeToBinary()
        self.basename = filebasename
        self.filecounter = -1
        self.writer.SetInput(self.vtkgrid)

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput (self.renWin)
        self.pngwriter = vtk.vtkPNGWriter()
        self.pngfilecounter = -1
        self.pngwriter.SetInput(w2if.GetOutput())
        self.w2if = w2if
        self.xmlfilecounter = -1

        try:
            self.pswriter = vtk.vtkGL2PSExporter()
            self.pswriter.SetRenderWindow(self.renWin)
            self.psfilecounter  = -1
            self.epsfilecounter = -1
            self.pdffilecounter = -1
        except:
            print "Could not import vtkGL2PSExporter from vtk. Saving pdf, eps, and ps files will not be possible."

        self.rawfilecounter = -1

    def write_xml(self, filename=None, map=None):
        if not self.is_writer: self.init_writer()
        self.xmlfilecounter += 1
        if filename == None:
            filename = "%s%.4d.xml" % (self.basename, self.xmlfilecounter)
        tf = open(filename, "w")
        tf.write("""\
<?xml version="1.0" encoding="UTF-8"?>

<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">
  <meshfunction type="double" dim="0" size="%d">
""" % len(self.x))
        x = self.x.copy()
        if map is not None:
            x[:] = x[map]
        for i,v in enumerate(x):
            tf.write('    <entity index="%d" value="%f"/>\n' % (i,v))
        tf.write("""\
  </meshfunction>
</dolfin>
""")
        tf.close()


    def write_vtk(self, filename=None):
        """Write data to file. This works because the attached field self.x is
        a pointer to the data that is being computed."""
        if not self.is_writer: self.init_writer()
        self.filecounter += 1
        if filename is None:
            filename = "%s%.4d.vtk" % (self.basename, self.filecounter)
        self._update()
        self.writer.SetFileName(filename)
        self.writer.Write()

    def write_png(self, filename=None):
        """Write a simulation frame to file in png format."""
        if not self.is_writer: self.init_writer()
        self.pngfilecounter += 1
        if filename is None:
            filename = "%s%.4d.png" % (self.basename, self.pngfilecounter)
        self._update()
        self.pngwriter.SetFileName(filename)
        self.w2if.Modified()
        self.pngwriter.Write()

    def write_ps(self, filename=None, compress = False, format = "eps"):
        """Write a simulation frame to file in eps/ps/pdf format."""
        if not self.is_writer: self.init_writer()
        assert(isinstance(format,str))
        if compress:
            self.pswriter.CompressOn()
        else:
            self.pswriter.CompressOff()
        if format.lower() == "pdf":
            self.pdffilecounter += 1
            self.pswriter.SetFileFormatToPDF()
            # Allways compress using pdf
            self.pswriter.CompressOn()
            counter = self.pdffilecounter
        elif format.lower() == "ps":
            self.psfilecounter += 1
            self.pswriter.SetFileFormatToPS()
            counter = self.psfilecounter
        else:
            self.epsfilecounter += 1
            self.pswriter.SetFileFormatToEPS()
            counter = self.epsfilecounter
        if filename is None:
            filename = "%s%.4d" % (self.basename, counter)
        self.pswriter.SetFilePrefix(filename)

        self._update()
        self.pswriter.Write()

    def write_raw(self, sol=None, dofmap=None):
        """Write a simulation frame to file, i.e. dump the numpy array to file
        in binary format."""
        if not self.is_writer: self.init_writer()
        #self._update(sol)
        self.rawfilecounter += 1

        if not dofmap is None:
            self.x[dofmap].dump("%s%.4d.raw" % (self.basename, self.rawfilecounter))
        else:
            self.x.dump("%s%.4d.raw" % (self.basename, self.rawfilecounter))

    def movie(self, name=None, fps=25, cleanup=False):
        import threading, subprocess

        class MyThread(threading.Thread):
            def __init__ (self, files, output, fps=10, cleanup=False):
                self.files = files
                self.output = output
                self.fps = fps
                self.cleanup = cleanup
                threading.Thread.__init__(self)

            def run(self):
                files = ",".join(self.files)
                #opts = "vbitrate=2160000:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:scplx_mask=0.1:tcplx_mask=0.1:naq"
                #command = 'mencoder mf://%s -mf fps=%d -o %s -ovc lavc -lavcopts vcodec=mpeg4:vpass=1:%s' % (files, self.fps, self.output, opts)
                #command = 'mencoder mf://%s -mf fps=%d -o %s -ovc lavc -lavcopts vcodec=mpeg4' % (files, self.fps, self.output)
                #command = 'mencoder mf://%s -mf fps=%d -o %s -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:mv0:trell=yes:v4mv=yes:cbp:last_pred=3:predia=2:dia=2:vmax_b_frames=2:vb_strategy=1:precmp=2:cmp=2:subcmp=2:preme=2:qns=2' % (files, self.fps, self.output)
                command = 'mencoder mf://%s -mf fps=%d -o %s -ovc lavc -lavcopts vcodec=ffv1' % (files, self.fps, self.output)
                failure = subprocess.call(command, shell=True)
                if failure:
                    print "Could not make video, please check your mencoder installation!"
                if self.cleanup:
                    for f in self.files:
                        os.remove(f)

        start = 0
        stop = self.pngfilecounter
        if stop < 0:
            print "No scenes written as png files. Giving up."
            return
        files = ["%s%.4d.png" % (self.basename, i) for i in xrange(start,stop+1)]
        if name == None:
            name = self.basename+".avi"
        mythread = MyThread(files, name, fps=fps, cleanup=cleanup)
        mythread.start()

    def interactive(self):
        """Hand the control over to the render window."""
        print "Plot active, press 'q' to continue."
        self.iren.Start()

    def azimuth(self, angle):
        self.ren.GetActiveCamera().Azimuth(angle)

    def elevate(self, x):
        self.ren.GetActiveCamera().Elevation(x)

    def dolly(self, x):
        self.ren.GetActiveCamera().Dolly(x)

    def set_viewangle(self, i):
        self.ren.GetActiveCamera().SetViewAngle(i)

    def set_camera_movement(self, a=0, e=0, m=60):
        self.azimuth_incr = a;   # 1 (degree) is a good value
        self.elevator_incr = e;  # 1 (degree) is a good value
        self.max_elevation = m;

    def _make_lut(self, autorange=(0,0)):
        lut = vtk.vtkLookupTable()
        lutfile = os.path.join(lutdir, self.lutfile)
        if os.path.isfile(self.lutfile):
            lutfile = self.lutfile
        if os.path.isfile(lutfile):
            vals = [x.split() for x in open(lutfile,'r').readlines()[1:]]
            lut.SetNumberOfColors(len(vals))
            lut.Build()
            for i in range(len(vals)):
                if len(vals[i]) == 4:
                    lut.SetTableValue(i, *[float(x) for x in vals[i]]),
        else:
            lut.SetTableRange (0, 1);
            lut.SetHueRange (0, 0);
            lut.SetSaturationRange (0, 0);
            lut.SetValueRange (0, 1);

        if False:
            ctable = [[1, 0, 0, 1],
                      [0, 1, 0, 1],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1],
                      [1, 1, 0, 1],
                      [1, 0, 1, 1],
                      [0, 1, 1, 1]
                      ]

            rmin, rmax = autorange
            ncols = int(rmax-rmin)+1
            lut.SetNumberOfColors(ncols)
            lut.Build()
            rnge = min(len(ctable), ncols)
            for i in xrange(rnge):
                row = ctable[i]
                lut.SetTableValue(i, *row)
        return lut

    def _make_scalarbar(self, lut):
        scalarbar = vtk.vtkScalarBarActor()
        scalarbar.SetLookupTable(lut)
        scalarbar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        scalarbar.GetPositionCoordinate().SetValue (0.1,0.01)
        scalarbar.SetOrientationToHorizontal()
        scalarbar.SetWidth (0.8)
        scalarbar.SetHeight (0.14)
        scalarbar.VisibilityOff()
        scalarbar.GetTitleTextProperty().SetColor(0,0,0)
        scalarbar.GetLabelTextProperty().SetColor(0,0,0)
        return scalarbar


    def make_vtk_grid(self, mesh, cell_type=None):
        """Based on a mesh with the methods cells() and coordinates(), construct a vtk grid."""
        #if mesh is None: mesh = self.mesh
        cell_dim = mesh.cells().shape[1]
        if cell_dim == 2:
            celltype = 3
        elif cell_dim == 3:
            celltype = 5
        elif cell_dim == 4:
            celltype = 10
        elif cell_type == "quads":
            celltype = 9

        cellsize = cell_dim + 1

        if cell_type == "quads":
            cellsize = 4

        nl = mesh.cells().copy()
        _cl = mesh.coordinates()
        cl = numpy.zeros((_cl.shape[0], 3), dtype='d')

        cl[:,:_cl.shape[1]] = _cl

        VTK_ID_TYPE_SIZE = vtk.vtkIdTypeArray().GetDataTypeSize()
        if VTK_ID_TYPE_SIZE == 4:
            inttype = numpy.int32
        elif VTK_ID_TYPE_SIZE == 8:
            inttype = numpy.int64

        # Create a cell array, stored in Python.numpy
        ncells = len(nl)
        cells = numpy.zeros((ncells, cellsize), dtype=inttype)

        cells[:,1:cellsize] = nl
        cells[:,0] = cellsize - 1

        a = numpy.ravel(cells)
        ita = vtk.vtkIdTypeArray()
        ita.SetVoidArray(a, len(a), 1)
        ca = vtk.vtkCellArray()
        ca.SetCells(ncells,ita)


        # Create some points
        npoints = len(cl)
        pts = numpy.ravel(cl)

        pa = vtk.vtkDoubleArray()
        pa.SetNumberOfComponents(3)
        pa.SetVoidArray(pts, npoints*3, 1)

        self.refs = [a, pts]

        v = vtk.vtkPoints()
        v.SetNumberOfPoints(npoints)
        v.SetData(pa)

        # Create an unstructured grid.
        us = vtk.vtkUnstructuredGrid()
        us.SetPoints(v)
        us.SetCells(celltype, ca)

        return us

    def vec3d(self, x):
        nsd = 3
        (n,m) = x.shape
        if m < nsd:
            x_old = x
            x = numpy.zeros((n, nsd))
            x[:,:m] = x_old
        return x


    def update(self, x=None):
        """Update plot data."""

        if self.displacement is not None:
            self.disp_arr.Modified()

        if not x is None:
            self._x_shape = x.shape
            self.x = x
            if len(self._x_shape) == 1:
                self.darr.SetVoidArray(self.x, len(self.x), 1)
            else:
                nsd = 3
                self.darr.SetNumberOfComponents(nsd)
                self.x = self.vec3d(self.x)
                if self.mode == "vector":
                    self._update_directions(self.x)
                self._x_shape=(nsd*len(self.x),)
                self.darr.SetVoidArray(self.x, len(numpy.ravel(self.x)), 1)
            #x.shape = shape
        if x is None and self.mode=="vector":
            self._update_directions(self.x)
        if x is None and self.mode=="scalar_xy":
            x = self.x

        self.darr.Modified()
        self.ren.ResetCameraClippingRange()
        if self.mesh is not None:
            self.vtkgrid.Modified()
            self.filter.Modified()

            if self.vertex_plot:
                if len(self._x_shape) == 1:
                    self.vtkgrid.GetPointData().SetScalars(self.darr)
                else:
                    self.vtkgrid.GetPointData().SetVectors(self.darr)
            else:
                if len(self._x_shape) == 1:
                    self.vtkgrid.GetCellData().SetScalars(self.darr)

        if self.mode == "scalar_xy":
            xVal = vtk.vtkFloatArray()
            yVal = vtk.vtkFloatArray()

            coords = self.mesh.coordinates()
            for i in range(len(coords)):
                xVal.InsertNextTuple1(coords[i])
                yVal.InsertNextTuple1(x[i])

            curve = vtk.vtkRectilinearGrid()
            curve.SetDimensions(len(coords), 1, 1)
            curve.SetXCoordinates(xVal)
            curve.GetPointData().SetScalars(yVal)

            xyplot = self.ren.GetViewProps().GetLastProp()
            xyplot.RemoveAllInputs()
            xyplot.AddInput(curve)

        if self.rescale and not self.mode == "scalar_xy":
            if self.mode == "vector":
                norms = [numpy.linalg.norm(v) for v in self.x]
                vmin, vmax = min(norms), max(norms)
            else:
                vmin, vmax = self.x.min(), self.x.max()
            self.rescaleColors(vmin, vmax)
        self.renWin.Render()

        # Set the window title after the window is actually created to
        # fix the problem with title not showing on OS X with Cocoa VTK
        # and on Windows.
        if self.frame is None:
            self.renWin.SetWindowName(self.title)

        self.azimuth(self.azimuth_incr)
        # assumes that initial elevation is zero. If not the sum of
        # elevate steps might exceed 90 degrees, which is not good.
        # Fix: use something like GetActiveCamera().getElevation()
        self.elevator += self.elevator_incr;
        step = 1.5*self.elevator_incr*math.cos(self.elevator*3.1416/180.0)
        self.elevate(step)

    def rescaleColors(self, dmin, dmax, idx=0):
        """Update's color scale """

        def _update_range(min, max, mapper):
            mapper.SetScalarRange(min, max)
            mapper.GetLookupTable().SetRange(min, max)
            self.scalarbar.GetLookupTable().SetRange(dmin, dmax)
            self.ren.ResetCamera()

        mapper = self.scalar_actors[idx].GetMapper()
        if mapper is not None:
            _update_range(dmin, dmax, mapper)
        mapper = self.vector_actors[idx].GetMapper()
        if mapper is not None:
            _update_range(dmin, dmax, mapper)
        self.show_scalarbar()

    def _add_arrow(self, vectors):
        tipradius = self.args.get("arrow_tip_radius", 0.15)
        tipresolution = self.args.get("arrow_tip_resolution", 16)
        tiplength= self.args.get("arrow_tip_length", 0.15)
        shaftradius = self.args.get("arrow_shaft_radius", 0.05)
        shaftresolution = self.args.get("arrow_shaft_resolution", 16)
        arrow = vtk.vtkArrowSource()
        arrow.SetTipRadius(tipradius)
        arrow.SetTipLength(tiplength)
        arrow.SetTipResolution(tipresolution)
        arrow.SetShaftRadius(shaftradius)
        arrow.SetShaftResolution(shaftresolution)
        vectors.SetSource(arrow.GetOutput())
        vectors.SetScaleFactor(0.01)
        return vectors

    def show_scalarbar(self):
        self.scalarbar.VisibilityOn()

    def set_vector_scale(self, scale_factor):
        """Set the vector scale factor."""
        self.vectors.SetScaleFactor(scale_factor)
        self.vectors.Modified()
        self.renWin.Render()

    def _update_directions(self, new_directions):
        self.directions[:,:] = new_directions
        if self.rescale:
            self.vectors.SetScaleFactor(self.compute_vector_scale(new_directions))
        self.vtkdirections.Modified()


    def _vectors_from_numpy(self, coords, directions):
        vectors = vtk.vtkGlyph3D()
        vectors.SetColorModeToColorByVector ()
        vectors.SetScaleModeToScaleByVector()
        coordinates = coords
        (n,m) = coordinates.shape
        coords_3d = numpy.zeros((n, 3), dtype='d')
        coords_3d[:,:m] = coordinates
        directions_3d = numpy.zeros((n, 3), dtype='d')
        if directions.shape[1] > m: # If directions is 3d already
            m = directions.shape[1]
        directions_3d[:,:m] = directions

        coordinates = coords_3d
        directions = directions_3d
        m = 3

        vtkpoints = vtk.vtkDoubleArray()
        vtkpoints.SetNumberOfComponents(m)
        vtkpoints.SetVoidArray(coordinates, n*m, m)
        _points = vtk.vtkPoints()
        _points.SetData(vtkpoints)

        vtkdirections = vtk.vtkDoubleArray()
        vtkdirections.SetNumberOfComponents(m)
        vtkdirections.SetVoidArray(directions, n*m, m)

        data = vtk.vtkPolyData()
        data.SetPoints(_points)
        data.GetPointData().SetVectors(vtkdirections)
        vectors.SetInput(data)
        vectors.SetVectorModeToUseVector()
        self.coordinates = coordinates
        self.directions = directions
        self.vtkpoints = vtkpoints
        self.vtkdirections = vtkdirections
        self.data = data
        return self._add_arrow(vectors)

    def _construct_interactor(self):
        if self.frame is not None:
            return None
        style = vtk.vtkInteractorStyleSwitch()
        style.SetCurrentStyleToTrackballCamera()
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(style)
        return iren

    def _construct_renderer(self):
        if self.frame is not None:
            from vtk.wx.wxVTKRenderWindow import wxVTKRenderWindow
            import wx
            size = self.frame.mainframe.GetClientSize()
            self.widget = wxVTKRenderWindow(self.frame, -1, size=size)
            ren = vtk.vtkRenderer()
            self.widget.GetRenderWindow().AddRenderer(ren)
            renWin = self.widget.GetRenderWindow()
        else:
            ren = vtk.vtkRenderer()
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(ren)
            renWin.SetSize(*self.window_size)
        return ren, renWin

    def compute_vector_scale(self, x, scale=2.0, vmax=None):
        n,d = x.shape
        if vmax is None:
            norms = [numpy.linalg.norm(v) for v in x]
            vmin, vmax = min(norms), max(norms)
            if abs(vmax) < 1e-16:
                vmax = 1.0
        return scale/vmax*self.__D/(n**(1.0/d))


    def vector_plotter(self, coords, x, vmin, vmax, wireframe=False, sf=2.0):
        import operator
        vectors = self._vectors_from_numpy(coords, x)

        if self.mesh is not None:
            a = self.simple_outline()[0].GetBounds()
        else:
            a = (0,1,0,1,0,1)
        self.__D = math.sqrt(reduce(operator.add,[(a[i]-a[i-1])**2 for i in xrange(1,6,2)]))

        # Resolve scale factor:
        if self.rescale:
            vectors.SetScaleFactor(self.compute_vector_scale(x, scale=sf))
        else:
            vectors.SetScaleFactor(self.compute_vector_scale(x, scale=sf, vmax=vmax))
        self.vectors = vectors

        if self.ren is None:
            self.ren, self.renWin = self._construct_renderer()
            self.iren = self._construct_interactor()
            if self.frame is None:
                self.iren.SetRenderWindow(self.renWin)
                self.iren.AddObserver("KeyPressEvent", self.key_press_methods)
            self.ren.SetBackground(1, 1, 1)

            lut = self._make_lut(autorange=(vmin, vmax))
            self.scalarbar = self._make_scalarbar(lut)
            self.lut = lut
        else:
            self.ren.RemoveAllViewProps()

        self.update_vector_mapper(vectors)
        self.rescaleColors(vmin, vmax)
        actor = self.vector_actors[-1] # TODO: Fixme
        actor.AddPosition(0, 0, 0)
        if self.axes_on:
            self.simple_axis(self.ren)

        self.ren.SetBackground(1, 1, 1)
        self.ren.AddActor(actor)
        self.ren.AddActor2D(self.scalarbar)
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Azimuth(00)
        self.ren.GetActiveCamera().Elevation(0)
        self.ren.GetActiveCamera().Dolly(1.5)
        self.ren.ResetCameraClippingRange()
        if self.frame is None:
            self.iren.AddObserver("KeyPressEvent", self.key_press_methods)
        return self.iren, self.renWin, self.ren

    def display_keybindings(self):
        print """\
Keybindings:

Viper specific:
* Keypress v: write data to file in vtk format
* Keypress i: write image to file in png format
* Keypress m: write image to file in eps format
* Keypress n: write image to file in pdf format
* Keypress o: add outline
* Keypress h: display this message

Inherited from vtk:
* Keypress j / Keypress t: toggle between joystick (position sensitive) and trackball (motion sensitive) styles. In joystick style, motion occurs continuously as long as a mouse button is pressed. In trackball style, motion occurs when the mouse button is pressed and the mouse pointer moves.

* Keypress c / Keypress a: toggle between camera and actor modes. In camera mode, mouse events affect the camera position and focal point. In actor mode, mouse events affect the actor that is under the mouse pointer.

* Button 1: rotate the camera around its focal point (if camera mode) or rotate the actor around its origin (if actor mode). The rotation is in the direction defined from the center of the renderer's viewport towards the mouse position. In joystick mode, the magnitude of the rotation is determined by the distance the mouse is from the center of the render window.

* Button 2: pan the camera (if camera mode) or translate the actor (if actor mode). In joystick mode, the direction of pan or translation is from the center of the viewport towards the mouse position. In trackball mode, the direction of motion is the direction the mouse moves. (Note: with 2-button mice, pan is defined as <Shift>-Button 1.)

* Button 3: zoom the camera (if camera mode) or scale the actor (if actor mode). Zoom in/increase scale if the mouse position is in the top half of the viewport; zoom out/decrease scale if the mouse position is in the bottom half. In joystick mode, the amount of zoom is controlled by the distance of the mouse pointer from the horizontal centerline of the window.

* Keypress 3: toggle the render window into and out of stereo mode. By default, red-blue stereo pairs are created. Some systems support Crystal Eyes LCD stereo glasses; you have to invoke SetStereoTypeToCrystalEyes() on the rendering window.

* Keypress e: exit the application.

* Keypress f: fly to the picked point

* Keypress p: perform a pick operation. The render window interactor has an internal instance of vtkCellPicker that it uses to pick.

* Keypress r: reset the camera view along the current view direction. Centers the actors and moves the camera so that all actors are visible.

* Keypress s: modify the representation of all actors so that they are surfaces.

* Keypress u: invoke the user-defined function. Typically, this keypress will bring up an interactor that you can type commands in.

* Keypress w: modify the representation of all actors so that they are wireframe.

* Keypress X: exit application
"""

    def simple_axis(self, ren):
        if self.mesh is not None:
            tprop = vtk.vtkTextProperty()
            tprop.SetColor(0, 0, 0)
            tprop.ShadowOff()
            outline = vtk.vtkOutlineFilter()
            outline.SetInput(self.vtkgrid)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(outline.GetOutputPort())
            axes = vtk.vtkCubeAxesActor2D()
            axes.SetInput(normals.GetOutput())
            axes.SetCamera(ren.GetActiveCamera())
            axes.GetProperty().SetColor(0,0,0)
            axes.SetAxisTitleTextProperty(tprop)
            axes.SetAxisLabelTextProperty(tprop)
            axes.SetCornerOffset(0)
            ren.AddViewProp(axes)

    def simple_outline(self):
        if self.mesh is not None:
            outline = vtk.vtkOutlineFilter()
            outline.SetInput(self.vtkgrid)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)
            return mapper, actor

    def key_press_methods(self, obj, event):
        key = obj.GetKeyCode()
        if key == "o":
            if not self.outline:
                map, actor = self.simple_outline()
                self.ren.AddActor(actor)
                self.renWin.Render()
                if True: # TODO: Figure out a way to steer this
                    bounds = map.GetBounds()
                    print "Size of bounding box:"
                    print "dx = ", bounds[1] - bounds[0]
                    print "dy = ", bounds[3] - bounds[2]
                    print "dz = ", bounds[5] - bounds[4]
                self.outline = True
        elif key == "i":
            print "Writing a simulation frame to a 'png' file"
            self.write_png()
        elif key == "v":
            print "Writing data to a 'vtk' file"
            self.write_vtk()
        elif key == "m":
            print "Writing a simulation frame to an 'eps' file"
            self.write_ps()
        elif key == "n":
            print "Writing a simulation frame to a 'pdf' file"
            self.write_ps(format="pdf")
        elif key == "h":
            self.display_keybindings()
        elif key == 'e':
            raise SystemExit

    def begin_interaction(self, obj, event):
        self.glyphActor.VisibilityOn()
        print "Picker activated"

    def enable_event(self, obj, event):
        self.glyphActor.VisibilityOn()
        print "Interaction is ON, press again to turn off."

    def disable_event(self, obj, event):
        self.glyphActor.VisibilityOff()
        print "Interaction is OFF, press again to turn on."

    def probe_data(self, obj, event):
        self.glyphActor.VisibilityOff()
        obj.GetPolyData(self.point)
        print obj.GetPosition()

    def user_method(self, obj, event):
        print "Please refrain form pressing 'u'"
        print self.pointWidget.GetPosition()
        self.add_sphere(self.pointWidget.GetPosition(), 0.2)
        self._update()

    def update_scalar_mapper(self, data, idx=0):
        """Set/Update mapper at position idx. Turn scalar data into a smooth vtkPolyDataNormals
        object for nice rendering."""
        extract = vtk.vtkGeometryFilter()
        extract.SetInput(data)
        extract.GetOutput().ReleaseDataFlagOn()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(extract.GetOutputPort())
        mapper = self.scalar_actors[idx].GetMapper()
        if mapper is None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(self.lut)
            self.scalar_actors[idx].SetMapper(mapper)

        mapper.SetInputConnection(normals.GetOutputPort())
        self.scalar_actors[idx].Modified()

    def update_vector_mapper(self, data, idx=0):
        """Set/Update vector mapper at position idx."""
        mapper = self.vector_actors[idx].GetMapper()
        if mapper is None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(self.lut)
            self.vector_actors[idx].SetMapper(mapper)
        mapper.SetInput(data.GetOutput())
        self.vector_actors[idx].Modified()

    def simple_plotter(self, data, vmin, vmax, wireframe=False):
        """Construct a simple vtk plotter for data."""

        if self.ren is None:
            self.ren, self.renWin = self._construct_renderer()
            self.iren = self._construct_interactor()
            if self.frame is None:
                self.iren.SetRenderWindow(self.renWin)
                self.iren.AddObserver("KeyPressEvent", self.key_press_methods)
            self.ren.SetBackground(1, 1, 1)

            lut = self._make_lut(autorange=(vmin, vmax))
            self.scalarbar = self._make_scalarbar(lut)
            self.lut = lut
        else:
            self.ren.RemoveAllViewProps()

        self.update_scalar_mapper(data)
        self.rescaleColors(vmin, vmax)
        actor = self.scalar_actors[-1] # TODO: Fixme
        actor.AddPosition(0, 0, 0)
        if wireframe:
            actor.GetProperty().SetRepresentationToWireframe ()

        # The plane widget is used probe the dataset.
        pointWidget = vtk.vtkPointWidget()
        pointWidget.SetInput(data)
        pointWidget.AllOff()
        pointWidget.PlaceWidget()
        point = vtk.vtkPolyData()
        pointWidget.GetPolyData(point)

        probe = vtk.vtkProbeFilter()
        probe.SetInput(point)
        probe.SetSource(data)

        # create glyph
        cone = vtk.vtkConeSource()
        cone.SetResolution(16)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInput(probe.GetOutput())
        glyph.SetSource(cone.GetOutput())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetScaleFactor(data.GetLength()*0.1)
        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.SetInput(glyph.GetOutput())
        glyphActor = vtk.vtkActor()
        glyphActor.SetMapper(glyphMapper)
        glyphActor.VisibilityOff()

        probe = vtk.vtkProbeFilter()
        probe.SetInput(point)
        probe.SetSource(data)

        self.cone = cone
        self.glyph = glyph
        self.probe = probe
        self.point = point
        self.glyphActor = glyphActor
        self.pointWidget = pointWidget

        self.sphere_data = vtk.vtkAppendPolyData()
        self.polygon_data = vtk.vtkAppendPolyData()
        self.smapper = vtk.vtkPolyDataMapper()

        for actor in self.sphere_actors:
            self.ren.AddActor(actor)

        for actor in self.scalar_actors:
            self.ren.AddActor(actor)

        self.ren.AddActor(glyphActor)
        if self.axes_on:
            self.simple_axis(self.ren)

        self.ren.AddActor2D(self.scalarbar)
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Azimuth(00)
        self.ren.GetActiveCamera().Elevation(0)
        self.ren.GetActiveCamera().Dolly(1.5)
        self.ren.ResetCameraClippingRange()

        if False:
            self.iren.AddObserver("UserEvent", self.user_method)
            pointWidget.SetInteractor(self.iren)
            pointWidget.AddObserver("EnableEvent", self.enable_event)
            pointWidget.AddObserver("DisableEvent", self.disable_event)
            pointWidget.AddObserver("StartInteractionEvent", self.begin_interaction)
            pointWidget.AddObserver("InteractionEvent", self.probe_data)

        if self.args.get("add_cell_labels", False):
            self.add_cell_labels()
        if self.args.get("add_point_labels", False):
            self.add_point_labels()

        return self.iren, self.renWin, self.ren


    def add_point_labels(self, labels=None):
        # Generate data arrays containing point and cell ids
        ldm = vtk.vtkLabeledDataMapper()
        ldm.SetLabelFormat("%g")
        ids = vtk.vtkIdFilter()
        if labels != None:
            print "Unable to plot custom point labels. Work in progress."
            """
            # Work in progress (?)
            ids.SetInput(self.vtkgrid)
            ids.PointIdsOn()
            ids.CellIdsOn()
            ids.FieldDataOn()
            ids.Update()
            tmp = ids.GetOutput()

            print "dir(tmp)", dir(tmp)
            print "dir(tmp.GetPointData())", dir(tmp.GetPointData())
            print "tmp.GetPointData().GetArrayName(idsname)", tmp.GetPointData().GetArray(0)
            print "dir(tmp.GetPointData().GetScalars())", dir(tmp.GetPointData().GetScalars())
            print "tmp", tmp
            arr = tmp.GetPointData().GetArray(0)
            arr.SetVoidArray(labels, len(labels), 1)
            #arr2 = tmp.GetPointData().GetScalars()
            #arr2.SetVoidArray(numpy.array(labels, dtype='d'), len(labels), 1)
            tmp.Modified()

            print "arr.GetName():", arr.GetName()
            n = arr.GetNumberOfTuples()
            print "n = ", n
            x = numpy.zeros(n, dtype='i')
            arr.ExportToVoidPointer(x)
            print "X:", x
            #tmp.Update()
            #tmp = self.vtkgrid
            """
        else:
            ids.SetInput(self.vtkgrid)
            ids.PointIdsOn()
            ids.FieldDataOn()
            tmp = ids.GetOutput()
            del ids

        ldm.SetInput(tmp)
        ldm.SetLabelModeToLabelFieldData()
        ldm.GetLabelTextProperty().SetColor(0, 0, 0)
        ldm.GetLabelTextProperty().BoldOn()
        pointLabels = vtk.vtkActor2D()
        pointLabels.SetMapper(ldm)
        self.ren.AddActor2D(pointLabels)

    def add_cell_labels(self):
        # Generate data arrays containing point and cell ids
        ids = vtk.vtkIdFilter()
        ids.SetInput(self.vtkgrid)
        ids.CellIdsOn()
        ids.FieldDataOn()
        cc = vtk.vtkCellCenters()
        cc.SetInputConnection(ids.GetOutputPort())
        tmp = cc.GetOutput()
        ldm = vtk.vtkLabeledDataMapper()
        ldm.SetLabelFormat("%g")
        ldm.SetInputConnection(cc.GetOutputPort())
        ldm.SetLabelModeToLabelFieldData()
        ldm.GetLabelTextProperty().SetColor(0, 0, 0)
        ldm.GetLabelTextProperty().BoldOn()
        pointLabels = vtk.vtkActor2D()
        pointLabels.SetMapper(ldm)
        self.ren.AddActor2D(pointLabels)


    def plot_xy(self, xd, yd, linespec, vmin=None, vmax=None):

        xVal = vtk.vtkFloatArray()
        yVal = vtk.vtkFloatArray()

        for i in range(len(xd)):
            xVal.InsertNextTuple1(xd[i])
            yVal.InsertNextTuple1(yd[i])

        curve = vtk.vtkRectilinearGrid()
        curve.SetDimensions(len(xd), 1, 1)
        curve.SetXCoordinates(xVal)
        curve.GetPointData().SetScalars(yVal)

        # Set up the xyplot actor
        xyplot = vtk.vtkXYPlotActor()

        text_prop = xyplot.GetTitleTextProperty()
        text_prop.SetColor(.0, .0, .0)
        text_prop.SetFontFamilyToArial()

        xyplot.AddInput(curve)
        xyplot.GetProperty().SetColor(0.0,0.0,0.0)
        xyplot.SetBorder(10)
        xyplot.GetPositionCoordinate().SetValue(0.0, 0.0, 0)
        xyplot.GetPosition2Coordinate().SetValue(1.0, 1.0, 0)
        xyplot.GetProperty().SetLineWidth(2)
        xyplot.GetProperty().SetPointSize(7)
        xyplot.SetPlotColor(0,1,0,0)
        xyplot.PlotPointsOff()
        xyplot.PlotLinesOff()
        xyplot.SetXTitle("x")
        xyplot.SetYTitle("u(x)")
        xyplot.SetXValuesToValue()
        xyplot.SetAxisTitleTextProperty(text_prop)
        xyplot.SetAxisLabelTextProperty(text_prop)
        xyplot.SetTitleTextProperty(text_prop)
        xyplot.SetDataObjectXComponent(0,0)
        xyplot.SetDataObjectYComponent(0,1)
        if not vmin is None and not vmax is None:
            xyplot.SetYRange(vmin, vmax)

        # determine line style
        if linespec == ".":
            xyplot.PlotCurvePointsOn()
            xyplot.SetPlotPoints(0,1)
        elif linespec == "-":
            xyplot.PlotCurveLinesOn()
        elif linespec == ".-" or linespec == "-.":
            xyplot.PlotCurvePointsOn()
            xyplot.PlotCurveLinesOn()
            xyplot.SetPlotPoints(0,1)
        else:
            print "Invalid line spec. Exiting"
            exit(0)

        # set up the renderer
        ren = vtk.vtkRenderer()
        ren.SetBackground(1,1,1)
        ren.AddActor(xyplot)

        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(*self.window_size)

        # interaction
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        return iren, renWin, ren

    def add_sphere(self, pt, rad, thera_res=8, phi_res=6, color=(0,0,0)):
        """Add a sphere at point pt, with radius rad."""
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(thera_res)
        sphere.SetPhiResolution(phi_res)
        sphere.SetRadius(rad)
        sphere.SetCenter(*pt)

        sactor = self.sphere_actors[-1]

        self.sphere_data.AddInput(sphere.GetOutput())
        self.smapper.SetInput(self.sphere_data.GetOutput())
        sactor.SetMapper(self.smapper)
        sactor.GetProperty().SetColor(*color)
        self.update()

    def add_polygon(self, polygon, idx=0):
        assert isinstance(polygon, (list, tuple))
        numpoints = len(polygon)
        assert isinstance(polygon[0], (list, tuple, numpy.ndarray))
        points2d = False
        if len(polygon[0]) == 2:
            points2d = True
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(numpoints)
        for i in xrange(numpoints):
            point = list(polygon[i])
            if points2d:
                point.append(0.0)
            points.InsertPoint(i, *point)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(numpoints)
        for i in xrange(numpoints):
            line.GetPointIds().SetId(i, i)
        grid = vtk.vtkUnstructuredGrid()
        grid.Allocate(1, 1)
        grid.InsertNextCell(line.GetCellType(),
                            line.GetPointIds())
        grid.SetPoints(points)

        extract = vtk.vtkGeometryFilter()
        extract.SetInput(grid)
        extract.GetOutput().ReleaseDataFlagOn()
        self.polygon_data.AddInput(extract.GetOutput())

        actor = self.polygon_actors[idx]
        mapper = actor.GetMapper()
        if mapper is None:
            mapper = vtk.vtkPolyDataMapper()
            actor.SetMapper(mapper)

        mapper.SetInput(self.polygon_data.GetOutput())
        actor.GetProperty().SetColor(0, 0, 1)
        actor.GetProperty().SetLineWidth(1)

        self.ren.AddActor(actor)
        self.update()

    def add_stim_sites(self, stim):
        """Add a sphere and a label to each stimulation site."""
        i = 0
        for item in stim:
            # draw a sphere at the stim site:
            point = item[0]
            if (len(point)==2):
                point =  (point[0], point[1], 0.0)
            self.add_sphere(point, item[1])
            # draw a lable at the stim site:
            act = vtk.vtkCaptionActor2D();
            act.SetCaption(str(i)); i += 1
            act.SetAttachmentPoint(point)
            act.BorderOff()
            act.GetProperty().SetColor(1, 0, 0)
            self.ren.AddActor(act)

    def _cutplane_filter(self, origo, normal):
        """ Return a cut plan data """
        if not self.mesh.cells().shape[1] == 4:
            raise RuntimeError, "Can only cut scalar 3 dimensional plots"

        if not (isinstance(origo, (numpy.ndarray, tuple, list)) and
                len(origo) == 3):
            raise RuntimeError, "Provide a tuple, list or numpy array of length 3 for cutplane_origo"

        if not (isinstance(normal, (numpy.ndarray, tuple, list)) and
                len(normal) ==3):
            raise RuntimeError, "Provide a tuple, list or numpy array of length 3 for cutplane_normal"

        # Define the cut plane
        plane = vtk.vtkPlane()
        plane.SetOrigin(origo)
        plane.SetNormal(normal)

        # Initialize the Cutter filter
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInput(self.filter)
        cut_data = cutter.GetOutput()
        return cut_data


    def set_sphere_opacity(self, val):
        """Change the sphere opacity (range [0,1])."""
        if val < 0.0: val = 0.0
        elif val > 1.0: val = 1.0

        for actor in self.sphere_actors():
            actor.GetProperty().SetOpacity(val)

    def set_min_max(self, min, max):
        """Set min and max scalar range"""
        self.rescaleColors(min, max)


def plot(mesh, data, *args, **kwargs):
    global _viper
    _viper = Viper(mesh, data, *args, **kwargs)
    return _viper

def update(data):
    global _viper
    if _viper != None:
        _viper.update(data)
        return _viper
    print "No plot object, cannot update"

def interactive():
    if _viper != None:
        _viper.interactive()
        return _viper
    print "No plot object, interaction not possible"

def save_plot(data, mesh, filename="plot.png"):
    return Viper(data, mesh, filename=filename, interactive=False)
