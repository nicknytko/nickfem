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
import pyamg
import pyamg.krylov

parser = argparse.ArgumentParser(
    prog='navier_img',
    description='Simulates a system with image applied as no-slip boundary conditions')
parser.add_argument('filename')
parser.add_argument('--invert', action='store_true')
parser.add_argument('--reynolds', type=float, default=100.0)
parser.add_argument('--inlet-velocity', type=float, default=0.01)
parser.add_argument('--contour', action='store_true')
parser.add_argument('--streamplot', action='store_true')
parser.add_argument('--streamplot-density', type=float, default=5.0)
parser.add_argument('--bicgstab', action='store_true', default=True, help='Use BiCGStab to solve Oseen equations')
parser.add_argument('--uzawa', action='store_true', help='Use Uzawa iteration to solve Oseen equations (broken)')
parser.add_argument('--plot-original', action='store_true', help='Plot original image under contours/streamplot')
parser.add_argument('--threshold', type=float, default=None,
                    help='Threshold (between 0 and 1, inclusive) for pixel values to be converted to boundaries')
parser.add_argument('--inlet-side', choices=['left', 'right', 'top', 'bottom'], default='left')
parser.add_argument('--outlet-side', choices=['left', 'right', 'top', 'bottom'], default='right')
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

aspect = img.shape[0] / img.shape[1]
xlim = (-1.0, 1.0)
ylim = (-aspect, aspect)

# Construct rectangular mesh lining up with image pixels
mesh = nickfem.Mesh.rectangular_mesh(img_grey.shape[1], img_grey.shape[0],
                                     xlim, ylim)
mesh2 = mesh.refine()
mesh2_lin = mesh2.to_linear()

x1, y1 = mesh.vertices.T
x2, y2 = mesh2.vertices.T

# Define boundary and interior nodes on the different spaces
bc1_sides = {
    'left': (x1 == xlim[0]),
    'right': (x1 == xlim[1]),
    'top': (y1 == ylim[1]),
    'bottom': (y1 == ylim[0])
}

bc2_sides = {
    'left': (x2 == xlim[0]),
    'right': (x2 == xlim[1]),
    'top': (y2 == ylim[1]),
    'bottom': (y2 == ylim[0])
}

boundary_nodes_1 = bc1_sides[args.outlet_side]
boundary_nodes_2 = ((bc2_sides['left'] | bc2_sides['right'] | bc2_sides['top'] | bc2_sides['bottom']) &
                    ~bc2_sides[args.inlet_side])

interp_img = sint.interpn((np.linspace(ylim[0], ylim[1], img_grey.shape[0]),
                           np.linspace(xlim[0], xlim[1], img_grey.shape[1])),
                          img_grey, np.column_stack((y2, x2)))

# Add boundary conditions from dark parts of the image
alpha = 0.9
threshold = np.mean(img_grey)
if args.threshold is not None:
    threshold = args.threshold

boundary_nodes_2 |= ((interp_img > threshold) &
                     ((x2 > alpha * xlim[0]) & (x2 < alpha * xlim[1]) &
                      (y2 > alpha * ylim[0]) & (y2 < alpha * ylim[1])))

int_nodes_1 = ~boundary_nodes_1
int_nodes_2 = ~boundary_nodes_2

# Dirichlet boundary conditions
def vel_x_bdy(x, y):
    u = np.zeros_like(x)
    u[x == xlim[0]] = args.inlet_velocity
    u[x == xlim[1]] = -args.inlet_velocity
    u[int_nodes_2] = 0.
    return u

def vel_y_bdy(x, y):
    u = np.zeros_like(x)
    u[y == ylim[0]] = args.inlet_velocity
    u[y == ylim[1]] = -args.inlet_velocity
    u[int_nodes_2] = 0.
    return u

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
print('Assembling bilinear operators')
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

# Solve Stokes equations for initial guess
print('Solving Stokes equations for initial guess')
Z = sp.block_array([
    [A, None, Dx],
    [None, A, Dy],
    [Dx.T, Dy.T, None]
])
u = R @ spla.bicgstab(R.T@Z@R, -R.T @ Z @ ub, x0=R.T@ub, rtol=1e-3)[0] + ub

eps = 2 / args.reynolds
u_hat = u.copy()
i = 0

def get_vp(u):
    up = u + ub
    N2 = len(x2)
    v_x = up[:N2]
    v_y = up[N2:2*N2]
    p = up[2*N2:]

    return (v_x, v_y, p)

def plot_soln(u):
    # Decompose solution into velocity and pressure components
    v_x, v_y, p = get_vp(u)

    # Plot norm of velocity
    v_norm = la.norm(np.array([v_x, v_y]), axis=0)

    plt.cla()

    if args.plot_original:
        plt.imshow(img_original, cmap='gray',
                   extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                   interpolation='bilinear')

    if args.contour or args.streamplot:
        N = mesh.num_vertices
        h, w = img_original.shape
        if args.contour:
            plt.contour(x1.reshape((h, w)), y1.reshape((h, w)),
                        v_norm[:N].reshape((h, w)), 16)
        else:
            plt.streamplot(x1.reshape((h, w)), y1.reshape((h, w)),
                           v_x[:N].reshape((h, w)),
                           v_y[:N].reshape((h, w)),
                           color=v_norm[:N].reshape((h, w)), density=args.streamplot_density, arrowsize=0)
    else:
        plt.tripcolor(x2, y2, v_norm, triangles=mesh2_lin.triangles, shading='gouraud')

    plt.axis('off')
    plt.show()
    plt.pause(0.5)

plt.ion()
plt.figure(figsize=(10, 10))
plt.show()
plt.axis('equal')
plot_soln(u)

def uzawa(C_u_hat, u, b, res_tol=1e-5):
    # doesn't work :-)
    D = sp.bmat([
        [R_2.T @ Dx @ R_1],
        [R_2.T @ Dy @ R_1]
    ])
    A_local = R_2.T @ (A * eps + C_u_hat) @ R_2
    A_ml = pyamg.smoothed_aggregation_solver(A_local, symmetry='nonsymmetric')
    N2 = len(x2)
    N2_r = R_2.shape[1]

    b1 = np.concatenate((
        R_2.T @ b[:N2],
        R_2.T @ b[N2:2*N2],
    ))
    b2 = R_1.T @ b[N2*2:]

    def A_full_matvec(u):
        return np.concatenate((
            A_local @ u[:N2_r],
            A_local @ u[N2_r:]
        ))

    def A_full_solve(b):
        b_x = b[:N2_r]
        b_y = b[N2_r:]
        return np.concatenate((
            A_ml.solve(b_x, accel='bicgstab'),
            A_ml.solve(b_y, accel='bicgstab')
        ))

    u2 = R_1.T @ u[N2*2:]
    u1 = A_full_solve(b1 - D @ u2)

    r2 = D.T @ u1 - b2
    p2 = r2

    for i in range(A.shape[0]):
        p1 = A_full_matvec(D@p2)
        a2 = D.T @ p1
        alpha = (p2@a2) / (p2@r2)
        u2 = u2 + alpha * p2
        r2 = r2 - alpha * a2
        u1 = u1 - alpha * p1

        beta = (r2@a2) / (p2@a2)
        p2 = r2 - beta * p2

        if la.norm(r2) / la.norm(b2) < res_tol:
            break

    return np.hstack([u1, u2])


while True:
    C_u_hat = assemble_conv_operator(u_hat)
    Z = sp.block_array([
        [A*eps + C_u_hat, None           , Dx],
        [None           , A*eps + C_u_hat, Dy],
        [Dx.T           , Dy.T           , None]
    ])

    sys = R.T @ Z @ R
    rhs = R.T @ (M @ u - Z @ ub)

    if args.uzawa:
        u_hat_new = R @ uzawa(C_u_hat, u_hat, M@u - Z@ub, 1e-8)
    elif args.bicgstab:
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

# Plot original solution and norm of velocity
plot_soln(u)

plt.figure()
v_x, v_y, p = get_vp(u)
plt.tripcolor(x2, y2, np.log10(1.0 + la.norm(np.array([v_x, v_y]), axis=0)),
              triangles=mesh2_lin.triangles, shading='gouraud')
plt.axis('equal')
plt.axis('off')

plt.show(block=True)
