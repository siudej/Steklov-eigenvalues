#!/usr/bin/python
"""
Main script for computing eigenvalues and geometric constants g_0 and g_1.

Set parameters below. Uncomment a starlike transformation to switch from
regular polygons to a star-like domain.
"""

from dolfin import set_log_active, parameters, FunctionSpace, Function, \
    FacetNormal, Expression, interpolate, TestFunction, FacetArea, assemble, \
    Constant, plot, avg, ds, dx, inner, dS, grad, pi
from solver import Solver, Regular, refine_perimeter, Starlike
import numpy as np
from numpy import cos, sin, sqrt
f = None
name = None

# FEM degree
degree = 2
# number of eigenvalues
numeig = 10
# sides for regular polygon
n = 5
# adaptive: try to find at least k eigenfunctions, use at most l (None for all)
k = 4
l = None
# max triangles
size = 400000
# starlike transformation, if needed
# (f, name) = ('1+0.8*cos(theta)', 'limacon0.8')
# (f, name) = ('sqrt(sin(theta)**2+0.01*cos(theta)**2)', 'hippopede0.01')
# (f, name) = ('sqrt(1-0.75)/sqrt(1-0.75*sin(theta)**2)', 'ellipse0.75')
# (f, name) = ('1', 'disk')
(f, name) = ('1-0.4*abs(cos(3*theta))', 'star2')


set_log_active(False)
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters["num_threads"] = 4

mesh = Regular(n)

# reshape mesh
if f is not None:
    star = Starlike(lambda theta: eval(f))
    mesh = star.shape(mesh)
    mesh.snap_boundary(star)
    mesh.smooth()

if name is None:
    name = 'regular' + str(n)
if k == 1:
    l = 1

print name
# plot mesh
pl = plot(mesh)
# create solver
solver = Solver(degree)

# adaptive refine
while mesh.size(2) < size / degree ** 2:
    print "Solving ..."
    sol = solver.solve(mesh, k)[:l]
    print "Eigenvalues :", sol[:, 0]  # print eigenvalues
    print "Refining ..."
    mesh = solver.adaptive(mesh, sol[:, 0], sol[:, 1])
    try:
        if f is not None and mesh.size(2) < 100000:
            print "Snapping ..."
            mesh = refine_perimeter(mesh)
            mesh.smooth()
            mesh.smooth_boundary()
            mesh.snap_boundary(star)
        print "Smoothing ..."
        mesh.smooth_boundary()
        mesh.smooth()
    except:
        print "Error"
    print "New mesh size: ", mesh.size(2)
    pl.plot(mesh)

# sizes of mesh triangles
V = FunctionSpace(mesh, "DG", 0)
W = FunctionSpace(mesh, "CG", 3)
v = TestFunction(V)
density = Function(V)
sides = FacetArea(mesh)
v = assemble(avg(v) * avg(sides) * dS + v * sides * ds)
density.vector()[:] = np.sqrt(v / np.median(v))
density = interpolate(density, W)

# perimeter
V = FunctionSpace(mesh, "R", 0)
u = Function(V)
u.interpolate(Constant(1.0))
L = assemble(u * ds)
print "Perimeter: ", L

# g0 and g1
n = FacetNormal(mesh)
xy = Expression(('x[0]', 'x[1]'))
g1 = assemble(inner(xy, xy) / inner(xy, n) * ds) * 2 * pi / L ** 2
g0 = assemble(1. / inner(xy, n) * ds) / 2 / pi
g = np.sqrt(g0 * g1)
print "g0, g1, g: ", g0, g1, g

# final solutions
print "Final solving ..."
sol = solver.solve(mesh, numeig)
print "Eigenvalues: "
eigs = np.zeros(len(sol[:, 0]))
for i in range(len(eigs)):
    # Rayleigh quotients
    eigs[i] = assemble(inner(grad(sol[i, 1]), grad(sol[i, 1]))
                       * dx) / assemble(sol[i, 1] * sol[i, 1] * ds)
    print sol[i, 0], eigs[i]

disk = np.cumsum(sorted(range(1, len(eigs)) * 2))
print "sum*L/(2pi)/disk (optimal g): "
sums = np.cumsum(eigs)
for i in range(len(eigs)):
    print sums[i] * L / 2 / pi / disk[i]

# make pdf from solutions
from export import export
export(name, mesh, eigs, sums, disk, sol, f, degree, n, k, l,
       size, L, g0, g1, g, density)
