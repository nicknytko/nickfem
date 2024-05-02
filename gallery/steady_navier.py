import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import nickfem
import matplotlib.pyplot as plt
import time

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

def mass_operator(args):
    u = args['u']
    v = args['v']
    wg = args['wg']

    # \int u v dx
    return np.einsum('ik, jk, k -> ij', u, v, wg)

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

def assemble_conv_operator(u):
    N2 = len(x2)
    u_x = u[:N2]
    u_y = u[N2:N2*2]

    def convection_operator(args):
        grad_u = args['grad_u']
        u = args['u']
        u_pts = args['u_tri']
        v = args['v']
        u_eval = np.row_stack(
            (np.einsum('i, ik -> k', u_x[u_pts], u),
             np.einsum('i, ik -> k', u_y[u_pts], u))
        )
        wg = args['wg']

        # \int (\hat{u} dot grad_u) v dx

        return np.einsum('jmk, mk, k, ik -> ij', grad_u, u_eval, wg, v)

    return nickfem.assemble_operator(mesh2, convection_operator, element_degree=(2, 2))

# Taylor-Hood discretization of the NS equations

A = nickfem.assemble_operator(mesh2, diffusion_operator, element_degree=(2, 2))
M_v = nickfem.assemble_operator(mesh2, mass_operator, element_degree=(2, 2))
M_p = nickfem.assemble_operator(mesh, mass_operator, element_degree=(1, 1))
M = sp.block_diag([M_v, M_v, M_p * 0])
Dx = nickfem.assemble_operator(mesh2, div_x_operator, element_degree=(2, 1))
Dy = nickfem.assemble_operator(mesh2, div_y_operator, element_degree=(2, 1))


# Restrictions from different spaces to interior of domain
R_1 = sp.eye(len(x1)).tocsc()[:, int_nodes_1]
R_2 = sp.eye(len(x2)).tocsc()[:, int_nodes_2]
R = sp.block_diag([R_2, R_2, R_1])

# Boundary conditions
ub = np.concatenate((vel_x_bdy(x2, y2), vel_y_bdy(x2, y2), p_bdy(x1, y1)))
u = ub.copy()

# Project out irrotational part of initial solution to obtain div-free field as initial u_hat
N2 = len(x2)
Dx2 = nickfem.assemble_operator(mesh2, div_x_operator, element_degree=(2, 2))
Dy2 = nickfem.assemble_operator(mesh2, div_y_operator, element_degree=(2, 2))
phi = spla.spsolve(A, Dx2@ub[:N2] + Dy2@ub[N2:2*N2])
u -= np.concatenate((Dx2@phi, Dy2@phi, np.zeros(len(x1))))

eps = 1/350

u_hat = u.copy()
i = 0
use_iterative = False

print('Performing Picard iteration...')

t_start = time.monotonic()

while True:
    C_u_hat = assemble_conv_operator(u_hat)
    Z = sp.block_array([
        [A*eps + C_u_hat, None           , Dx],
        [None           , A*eps + C_u_hat, Dy],
        [Dx.T           , Dy.T           , None]
    ])

    sys = R.T @ Z @ R
    rhs = R.T @ (M @ u - Z @ ub)

    if use_iterative:
        u_hat_new = R @ spla.bicgstab(sys, rhs, x0=R.T@u_hat, rtol=1e-6)
    else:
        u_hat_new = R @ spla.spsolve(sys, rhs)
    diff = la.norm(u_hat_new - u_hat)
    u_hat = u_hat_new

    print(f'{i:<2}: L2 diff {diff:.3e}')

    i += 1
    if diff < 1e-5:
        u = u_hat
        break

t_end = time.monotonic()
print(f'Elapsed time: {t_end-t_start:.3f} seconds')

# Decompose solution into velocity and pressure components
u_plot = u + ub
N2 = len(x2)
v_x = u_plot[:N2]
v_y = u_plot[N2:2*N2]
p = u_plot[2*N2:]

# Plot norm of velocity
v_norm = la.norm(u_plot[:2*N2].reshape((2, -1)), axis=0)
fig = ff.create_trisurf(x=x2, y=y2, z=v_norm,
                         simplices=mesh2_lin.triangles)
fig.show()
