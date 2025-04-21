import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import nickfem
from nickfem.misc import Timer
import matplotlib.pyplot as plt
import pyamg

N = 128
x_ = np.linspace(-1, 1, N)
y_ = np.linspace(-1, 1, N)

x, y = np.meshgrid(x_, y_)
x = x.flatten()
y = y.flatten()

mesh = nickfem.Mesh(np.column_stack((x, y)))
mesh2 = mesh.refine()
mesh2_lin = mesh2.to_linear()

x2, y2 = mesh2.vertices.T

interior_nodes = ((x2 > -1) & (x2 < 1) & (y2 > -1) & (y2 < 1))
boundary_nodes = ~interior_nodes

R_int = sp.eye(len(x2)).tocsc()[:, interior_nodes]
R_bdy = sp.eye(len(x2)).tocsc()[:, boundary_nodes]

def dirichlet_bdy(x, y):
    u = np.zeros_like(x)
    u[x==1] = 1.
    return u

def wind(x, y):
    return np.column_stack((2 * y * (1-x**2), -2 * x * (1 - y**2)))

# Galerkin formulation of operators

def diffusion_operator(args):
    grad_u = args['grad_u']
    grad_v = args['grad_v']
    wg = args['wg']

    # \int grad_u dot grad_v dx

    return np.einsum('ilk, jlk, k -> ij', grad_u, grad_v, wg)

def convection_operator(args):
    grad_u = args['grad_u']
    v = args['v']
    w_eval = wind(args['xg_global'], args['yg_global']).T
    wg = args['wg']

    # \int (w dot grad_u) v dx

    return np.einsum('jmk, mk, k, ik -> ij', grad_u, w_eval, wg, v)

with Timer('Assembly'):
    A = nickfem.assemble_operator(mesh2, diffusion_operator)
    C = nickfem.assemble_operator(mesh2, convection_operator)
    Z = A / 200 + C

with Timer('MG Setup'):
    ml = pyamg.smoothed_aggregation_solver(R_int.T@Z@R_int, symmetry='nonsymmetric')

with Timer('MG Solve (BiCGStab)'):
    ub = dirichlet_bdy(x2, y2)
    f = -Z @ ub

    u = R_int @ ml.solve(R_int.T@f, accel='bicgstab') + ub

import plotly.graph_objects as go
import plotly.figure_factory as ff

fig = ff.create_trisurf(x=x2, y=y2, z=u,
                         simplices=mesh2_lin.triangles,
                         aspectratio=dict(x=1, y=1, z=1.0))
fig.show()
