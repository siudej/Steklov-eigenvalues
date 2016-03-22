""" FEniCS 1.6 solver implementation. """
from dolfin import FunctionSpace, TrialFunction, TestFunction, inner, grad, \
    ds, dx, assemble, SLEPcEigenSolver, Function, PETScMatrix, CellSize, \
    FacetNormal, CellFunction, avg, jump, Constant, div, sqrt, refine, dS, \
    cells, facets, Point, SubDomain, BoundaryMesh, vertices, \
    edges
from mshr import Polygon, generate_mesh
import numpy as np
import numpy.linalg as nl


class Solver:

    """Adaptive solver class. """

    def __init__(self, degree):
        """ Save FEM degree. """
        self.degree = degree

    def solve(self, mesh, num=5):
        """ Solve for num eigenvalues based on the mesh. """
        # conforming elements
        V = FunctionSpace(mesh, "CG", self.degree)
        u = TrialFunction(V)
        v = TestFunction(V)

        # weak formulation
        a = inner(grad(u), grad(v)) * dx
        b = u * v * ds
        A = PETScMatrix()
        B = PETScMatrix()
        A = assemble(a, tensor=A)
        B = assemble(b, tensor=B)

        # find eigenvalues
        eigensolver = SLEPcEigenSolver(A, B)
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["spectrum"] = "smallest real"
        eigensolver.parameters["spectral_shift"] = 1.0E-10
        eigensolver.solve(num + 1)

        # extract solutions
        lst = [
            eigensolver.get_eigenpair(i) for i in range(
                1,
                eigensolver.get_number_converged())]
        for k in range(len(lst)):
            u = Function(V)
            u.vector()[:] = lst[k][2]
            lst[k] = (lst[k][0], u)  # pair (eigenvalue,eigenfunction)
        return np.array(lst)

    def adaptive(self, mesh, eigv, eigf):
        """Refine mesh based on residual errors."""
        fraction = 0.1
        C = FunctionSpace(mesh, "DG", 0)  # constants on triangles
        w = TestFunction(C)
        h = CellSize(mesh)
        n = FacetNormal(mesh)
        marker = CellFunction("bool", mesh)
        print len(marker)
        indicators = np.zeros(len(marker))
        for e, u in zip(eigv, eigf):
            errform = avg(h) * jump(grad(u), n) ** 2 * avg(w) * dS \
                + h * (inner(grad(u), n) - Constant(e) * u) ** 2 * w * ds
            if self.degree > 1:
                errform += h ** 2 * div(grad(u)) ** 2 * w * dx
            indicators[:] += assemble(errform).array()  # errors for each cell
        print "Residual error: ", sqrt(sum(indicators) / len(eigv))
        cutoff = sorted(
            indicators, reverse=True)[
            int(len(indicators) * fraction) - 1]
        marker.array()[:] = indicators > cutoff  # mark worst errors
        mesh = refine(mesh, marker)
        return mesh

# domains


def Regular(n, size=50):
    """Build mesh for a regular polygon with n sides."""
    points = [(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)]
    polygon = Polygon([Point(*p) for p in points])
    return generate_mesh(polygon, size)


def refine_perimeter(mesh):
    """Refine largest boundary triangles."""
    mesh.init(1, 2)
    perimeter = [c for c in cells(mesh)
                 if any([f.exterior() for f in facets(c)])]
    marker = CellFunction('bool', mesh, False)
    max_size = max([c.diameter() for c in perimeter])
    for c in perimeter:
        marker[c] = c.diameter() > 0.75 * max_size
    return refine(mesh, marker)


class Starlike(SubDomain):

    """Class for building a mesh from a radius function."""

    def __init__(self, f):
        """f should be a Python function with angle as argument."""
        self.f = f
        super(Starlike, self).__init__()

    def snap(self, x):
        """Make sure boundary vertices are on theoretical boundary."""
        x[:] *= self.f(np.arctan2(x[1], x[0])) / nl.norm(x)

    def shape(self, mesh, size=50):
        """Build mesh."""
        vf = np.vectorize(self.f)
        x = mesh.coordinates()[:, 0]
        y = mesh.coordinates()[:, 1]
        a = np.arctan2(y, x)
        x, y = [x * vf(a), y * vf(a)]
        mesh.coordinates()[:] = np.array([x, y]).transpose()
        boundary = BoundaryMesh(mesh, 'exterior')
        boundary.init()
        lst = [0]
        vs = list(vertices(boundary))
        while True:
            v = vs[lst[-1]]
            neighbors = set()
            for e in edges(v):
                neighbors.update(e.entities(0))
            neighbors.remove(v.index())
            neighbors = list(neighbors)
            k = 0
            if len(lst) > 1:
                if neighbors[0] == lst[-2]:
                    k = 1
            lst.append(neighbors[k])
            if lst[-1] == lst[0]:
                break
        lst = lst[:-1]
        points = boundary.coordinates()[lst]
        points = [Point(*p) for p in points]
        try:
            polygon = Polygon(points)
        except:
            polygon = Polygon(points[::-1])
        return generate_mesh(polygon, size)
