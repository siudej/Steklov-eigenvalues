""" Export computations to a tex file and run latex. """
import viper3d as vp
import os
from numpy import pi
import sympy


def export(name, mesh, eigs, sums, disk, sol, f, degree, n, k, l,
           size, L, g0, g1, g, density):
    """ Generate and compile a Latex report. """
    try:
        os.mkdir('reports')
    except:
        pass
    os.chdir('reports')
    for i in range(len(eigs)):
        vi = vp.Viper3D(mesh, sol[i, 1].compute_vertex_values(mesh))
        vi.ContourPlot(10)  # plot contours in 2D
        fullname = name + '_u{:02d}.png'.format(i + 1)
        vi.write_png(fullname)
        os.system('convert {0} -trim {0}'.format(fullname))  # crop image
    vi = vp.Viper3D(mesh, density.compute_vertex_values(mesh))
    vi.Plot(opacity=0.5)  # plot triangle sizes
    fullname = name + '_density.png'
    vi.write_png(fullname)
    os.system('convert {0} -trim {0}'.format(fullname))  # crop image
    fullname = name + 'd{}'.format(degree) + 's{}'.format(size) + '.tex'
    fl = open(fullname, "w")
    if f is not None:
        domain = r"""
Radius function:
\begin{align*}
R(\theta)=""" + sympy.latex(sympy.sympify(f)) + r"""
\end{align*}"""
    else:
        domain = r"Regular polygon with {} sides.".format(n)
    fl.write(r"""
\documentclass{amsart}
\usepackage{tikz}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}

\begin{document}

\section{Eigenvalues for the domain """ + name + "}" + domain + r"""

Adaptive finite element method of degree {} with {} triangles.""".format(degree, mesh.size(2)) + r"""
\subsection{Geometric quantities (perimeter $L$ and constants $g_i$)}
All quantities evaluated using boundary integral for a very fine mesh. For $g_i$ we use geometric representations from Lemma 6.2.
\begin{align*}
""" + r"L={}\quad g_0={}\quad g_1={}\quad g={}".format(L, g0, g1, g) + r"""
\end{align*}

\subsection{Eigenvalues}
Upper bounds are obtained by plugging numerical eigenfunctions into the Rayleigh quotient. Numbers in parentheses are numerical eigenvalues of the discrete problem. Finally $\rho_i$ are the rescaled sums $(\sigma_1+\cdots+\sigma_n)L$ on the domain divided by the same quantity on the disk.
\begin{align*}""" + r"\\".join([r"\sigma_{{{}}}&\le{}\quad&({})\qquad&\qquad&\qquad\rho_{{{}}}&={}".format(i + 1, eigs[i], sol[i, 0], i + 1, sums[i] * L / 2 / pi / disk[i]) for i in range(len(eigs))]) + r"""
\end{align*}

\newpage
\subsection{Plots of eigenfunctions}$\;$

\noindent
""" + "\n".join([r"\includegraphics[width=0.33\textwidth]{}".format("{{{" + name + '_u{:02d}'.format(i + 1) + '}}} ') for i in range(len(eigs))]) + r"""
\newpage
\subsection{Sizes of mesh triangles and list of other parameters.}$\;$

""" + r"Initial polygon: {} sides. Adaptive: try to find {} eigenfunctions, use at most {}. Refine to at least {} triangles.".format(n, k, l, size) + r"""

Sizes of mesh triangles after adaptive refinement (blue - small):

\includegraphics[width=0.33\textwidth]{{{""" + name + r"""_density}}}
\end{document}
""")
    fl.close()
    os.system('pdflatex --interaction=batchmode ' + fullname)
    os.system('pdflatex --interaction=batchmode ' + fullname)
    os.system('pdflatex --interaction=batchmode ' + fullname)
    os.system('rm *.log')
    os.system('rm *.aux')
    os.system('rm *.tex')
    os.system('rm *.png')
