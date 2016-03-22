# Bounds for Steklov eigenvalues

This is a collection of scripts for computing bounds and numerical solutions for Steklov eigenvalues on regular polygons and quasi-conformally transformed disks.

These were developed for the following paper I coauthored with Alexandre Girouard and Richard Laugesen

### [Steklov Eigenvalues and Quasiconformal Maps of Simply Connected Planar Domains](http://link.springer.com/article/10.1007%2Fs00205-015-0912-8)

Running the main `solve.py` script produces a LaTeX file with the numerical data and images. The compiled reports are placed in the `reports` folder, and a few examples are included.

### Numerical method

We used [FEniCS](http://fenicsproject.org) to implement the finite element scheme. We employ a two-fold adaptive mesh refinement approach. We subdivide the mesh triangles based on residual errors in the solutions: derivative jumps on edges, PDE errors inside triangles, and inexact eigenvalue conditions on the boundary of the domain. We also use FEniCS's domain snapping feature to refine the trinagles on the boundary of the domain, to improve polygonal approximation of the smooth curved boundaries.

### Examples

The first image below shows the density of the adaptively refined triangulation for a star shaped domain (blue - many small triangles). One can notice a lot of refinements near the negatively curved boundary. The reentrant corners in the polygonal approximation lead to the loss of smoothness of the eigenfunctions and decreased FEM accuracy. Hence the intense local refinements.

<img src="star_density.png" height="400"/> <img src="hippopede0.01_density.png" height="400"/>

The hippopede shaped region shows frequent refinements near the narrow part of the domain (reason as above), as well as many small triangles near the round boundary (shape approximation improvements).

Finally, the two images below are examples of the computed eigenvalues.

<img src="regular5.png" height="400"/> <img src="hippopede0.01.png" height="400"/>
