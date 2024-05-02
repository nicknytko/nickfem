import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import nickfem
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.figure_factory as ff

mesh = nickfem.Mesh.load('../mesh/L.msh')
mesh2 = mesh.refine()
mesh2_lin = mesh2.to_linear()

x1, y1 = mesh.vertices.T
x2, y2 = mesh2.vertices.T

boundary_nodes_1 = (y1 == -3) # Pressure dirichlet on outlet
boundary_nodes_2 = ((x2 == 0) | (y2 == 1) | (x2 == 6) | ((y2 == 0) & (x2 <= 5)) | ((x2 == 5) & (y2 <= 0)))

int_nodes_1 = ~boundary_nodes_1
int_nodes_2 = ~boundary_nodes_2

def vel_x_bdy(x, y):
    u = np.zeros_like(x)
    u[x == 0] = 1.
    return u

def vel_y_bdy(x, y):
    return np.zeros_like(x)

def p_bdy(x, y):
    return np.zeros_like(x)

# Galerkin formulation of operators

def diffusion_operator(args):
    grad_u = args['grad_u']
    grad_v = args['grad_v']
    wg = args['wg']

    # \int grad_u dot grad_v dx

    return np.einsum('ilk, jlk, k -> ij', grad_u, grad_v, wg)

def div_x_operator(args):
    grad_u = args['grad_u']
    v = args['v']
    wg = args['wg']

    # - \int du/dx v dx

    return - np.einsum('ik, jk, k -> ij', grad_u[:, 0, :], v, wg)

def div_y_operator(args):
    grad_u = args['grad_u']
    v = args['v']
    wg = args['wg']

    # - \int du/dy v dx

    return - np.einsum('ik, jk, k -> ij', grad_u[:, 1, :], v, wg)

# Taylor-Hood discretization of the Stokes equations

A = nickfem.assemble_operator(mesh2, diffusion_operator, element_degree=(2, 2))
Dx = nickfem.assemble_operator(mesh2, div_x_operator, element_degree=(2, 1))
Dy = nickfem.assemble_operator(mesh2, div_y_operator, element_degree=(2, 1))

Z = sp.block_array([
    [A, None, Dx],
    [None, A, Dy],
    [Dx.T, Dy.T, None]
])

# Restrictions from different spaces to interior of domain
R_1 = sp.eye(len(x1)).tocsc()[:, int_nodes_1]
R_2 = sp.eye(len(x2)).tocsc()[:, int_nodes_2]
R = sp.block_diag([R_2, R_2, R_1])

# Boundary conditions
ub = np.concatenate((vel_x_bdy(x2, y2), vel_y_bdy(x2, y2), p_bdy(x1, y1)))
sol = R @ spla.spsolve(R.T @ Z @ R, -R.T @ Z @ ub) + ub

# Decompose solution into velocity and pressure components
N2 = len(x2)
vel_x = sol[:N2]
vel_y = sol[N2:2*N2]
p = sol[2*N2:]

# Plot norm of velocity
vel_norm = la.norm(sol[:2*N2].reshape((2, -1)), axis=0)
fig = ff.create_trisurf(x=x2, y=y2, z=vel_norm,
                         simplices=mesh2_lin.triangles)
fig.show()
