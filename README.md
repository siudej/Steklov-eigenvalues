# Bounds for Steklov eigenvalues

This is a collection of scripts for computing bounds for Steklov eigenvalues on regular polygons and quasi-conformally transformed disks.

These were developed for the following paper I coauthored with Alexandre Girouard and Richard Laugesen

### [Steklov Eigenvalues and Quasiconformal Maps of Simply Connected Planar Domains](http://link.springer.com/article/10.1007%2Fs00205-015-0912-8)

Running the main `solve.py` script produces a LaTeX file with the numerical data and images. The compiled reports are placed in the `reports` folder, and a few examples are included.

### Numerical method

We used [FEniCS](http://fenicsproject.org) to implement the finite element scheme. We employ a two-fold adaptive mesh refinement approach. We subdivide the mesh triangles based on residual errors in the solutions: derivative jumps on edges, PDE errors inside triangles, and inexact eigenvalue conditions on the boundary of the domain. We also use FEniCS's domain snapping feature to refine the trinagles on the boundary of the domain, to improve polygonal approximation of the smooth curved boundaries.
