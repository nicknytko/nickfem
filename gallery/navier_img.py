import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.interpolate as sint
import nickfem
import matplotlib.pyplot as plt
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
import argparse

parser = argparse.ArgumentParser(
    prog='navier_img',
    description='Simulates a system with image applied as pressure forcing term')
parser.add_argument('filename')
parser.add_argument('--invert', action='store_true', default=False)
parser.add_argument('--reynolds', type=float, default=10.0)
parser.add_argument('--inlet-velocity', type=float, default=1.0)
parser.add_argument('--enclosed', action='store_true', default=True)
args = parser.parse_args()

# Read image, normalize between 0 and 1, and convert to greyscale
img = imread(args.filename).astype(np.float64)
img -= np.min(img)
img /= np.max(img)
img_grey = np.mean(img, axis=2)
img_original = img_grey.copy()

img_grey = img_grey[::-1, :]
if args.invert:
    img_grey = 1.0 - img_grey

# Construct rectangular mesh lining up with image pixels
mesh = nickfem.Mesh.rectangular_mesh(img_grey.shape[1], img_grey.shape[0])
mesh2 = mesh.refine()
mesh2_lin = mesh2.to_linear()

x1, y1 = mesh.vertices.T
x2, y2 = mesh2.vertices.T

# Define boundary and interior nodes on the different spaces
if args.enclosed:
    boundary_nodes_1 = (x1 == 1)
    boundary_nodes_2 = (x2 == -1) | (y2 == -1) | (y2 == 1)

    interp_img = sint.interpn((np.linspace(1, -1, img_grey.shape[0]),
                               np.linspace(-1, 1, img_grey.shape[1])),
                              img_grey, np.column_stack((y2, x2)))
    boundary_nodes_2 |= ((interp_img > np.mean(img_grey)) &
                         ((x2 > -0.9) & (x2 < 0.9) & (y2 > -0.9) & (y2 < 0.9)))
else:
    boundary_nodes_1 = (x1 == 1) | (y1 == -1) | (y1 == 1)
    boundary_nodes_2 = (x2 == -1)

int_nodes_1 = ~boundary_nodes_1
int_nodes_2 = ~boundary_nodes_2

# Dirichlet boundary conditions
def vel_x_bdy(x, y):
    u = np.zeros_like(x)
    u[x == -1] = args.inlet_velocity
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

# Forcing term
f = np.concatenate((np.zeros(len(x2)), np.zeros(len(x2)), M_p @ img_grey.flatten()))

eps = 1 / args.reynolds
u_hat = u.copy()
i = 0
use_iterative = True

def plot_soln(u):
    # Decompose solution into velocity and pressure components
    u_plot = u + ub
    N2 = len(x2)
    v_x = u_plot[:N2]
    v_y = u_plot[N2:2*N2]
    p = u_plot[2*N2:]

    # Plot norm of velocity
    v_norm = la.norm(u_plot[:2*N2].reshape((2, -1)), axis=0)

    plt.cla()
    #plt.tripcolor(x2, y2, v_norm, triangles=mesh2_lin.triangles, shading='gouraud')
    plt.imshow(img_original, cmap='gray', extent=(-1, 1, 1, -1))
    N = mesh.num_vertices
    h, w = img_original.shape
    plt.contour(x1.reshape((h, w)), y1.reshape((h, w)),
                v_norm[:N].reshape((h, w)), 32)
    plt.show()
    plt.pause(0.5)

plt.ion()
plt.show()
plt.axis('equal')
plot_soln(u)

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
        u_hat_new = R @ spla.bicgstab(sys, rhs, x0=R.T@u_hat, rtol=1e-3)[0]
    else:
        u_hat_new = R @ spla.spsolve(sys, rhs)
    diff = la.norm(u_hat_new - u_hat)
    u_hat = u_hat_new

    print(f'{i:<2}: L2 diff {diff:.3e}')
    plot_soln(u_hat)

    i += 1
    if diff < 1e-5:
        u = u_hat
        break

t_end = time.monotonic()
print(f'Elapsed time: {t_end-t_start:.3f} seconds')
