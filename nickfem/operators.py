import numpy as np

def mass(args):
    u = args['u']
    v = args['v']
    wg = args['wg']

    # \int u dot v dx

    return np.einsum('ik, jk, k -> ij', u, v, wg)

def diffusion(args):
    grad_u = args['grad_u']
    grad_v = args['grad_v']
    wg = args['wg']

    # \int grad_u dot grad_v dx

    return np.einsum('ilk, jlk, k -> ij', grad_u, grad_v, wg)

def advection(wind_fn):
    def op(args):
        grad_u = args['grad_u']
        v = args['v']
        w_eval = wind_fn(args['xg_global'], args['yg_global']).T
        wg = args['wg']

        # \int (w dot grad_u) v dx

        return np.einsum('jmk, mk, k, ik -> ij', grad_u, w_eval, wg, v)
    return op

def divergence_x(args):
    grad_u = args['grad_u']
    v = args['v']
    wg = args['wg']

    # - \int du/dx v dx

    return - np.einsum('ik, jk, k -> ij', grad_u[:, 0, :], v, wg)

def divergence_y(args):
    grad_u = args['grad_u']
    v = args['v']
    wg = args['wg']

    # - \int du/dy v dx

    return - np.einsum('ik, jk, k -> ij', grad_u[:, 1, :], v, wg)
