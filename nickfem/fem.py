import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from scipy.spatial import Delaunay
import nickfem.gausstri as gausstri
import meshio


class Mesh:
    """Triangular mesh"""

    def __init__(self, vertices, triangles=None, degree=1):
        """
        Instantiates a triangle mesh object.  Currently supports
        only linear or quadratic triangles.

        Parameters
        ----------
        vertices : numpy.ndarray of size (num_verts, 2)
          Vertices of the triangles
        triangles : numpy.ndarray of size (num_tris, ...)
          Integer pointers to vertices.  Each triangle should have
          either 3 entries (linear) or 6 entries (quadratic)
        """

        self.vertices = vertices
        self.num_vertices = vertices.shape[0]
        if triangles is None:
            triangles = Delaunay(vertices).simplices
        self.triangles = triangles
        self.num_triangles = triangles.shape[0]
        self.degree = degree

        # Used to keep track of number of vertices for mixed-space discretizations
        # like for discrete divergence operator
        self.degree_verts = {
            degree: self.num_vertices
        }

    def load(fname):
        msh = meshio.read(fname)
        verts = msh.points[:, :2]
        tris = None

        for cell in msh.cells:
            if cell.type == 'triangle':
                tris = cell.data
                break

        if tris is None:
            raise RuntimeError('Could not find "triangles" cell in mesh data')

        return Mesh(verts, tris, degree=1)

    def rectangular_mesh(logical_width, logical_height, x_lim=(-1, 1), y_lim=(-1, 1)):
        """
        Creates a rectangular triangle mesh.

        Parameters
        ----------
        logical_width : integer
          Number of vertices in the horizontal direction
        logical height : integer
          Number of vertices in the vertical direction
        x_lim : tuple
          X limits for the physical domain
        y_lim : tuple
          Y limits for the physical domain

        Returns
        -------
        nickfem.Mesh object
        """

        x_ = np.linspace(x_lim[0], x_lim[1], logical_width)
        y_ = np.linspace(y_lim[0], y_lim[1], logical_height)

        x, y = np.meshgrid(x_, y_, indexing='xy')
        x = x.flatten()
        y = y.flatten()
        triangles = []

        def ij_to_idx(i, j):
            return j * logical_width + i

        for i in range(logical_width - 1):
            for j in range(logical_height - 1):
                # First triangle:
                # i,j
                # |     \
                # i,j+1 --- i+1,j+1

                triangles.append(np.array([
                    ij_to_idx(i, j),
                    ij_to_idx(i, j+1),
                    ij_to_idx(i+1, j+1)
                ]))

                # Second triangle:
                # i,j   --- i+1, j
                #        \    |
                #           i+1,j+1

                triangles.append(np.array([
                    ij_to_idx(i, j),
                    ij_to_idx(i+1, j+1),
                    ij_to_idx(i+1, j)
                ]))
        triangles = np.array(triangles)

        return Mesh(np.column_stack((x, y)), triangles)


    def refine(self):
        """
        Refines a linear triangular mesh to quadratic triangles by
        splitting unique edges

        New triangles are ordered like:
        3
        | \
        4   6
        |     \
        1 - 5 - 2
        """

        assert(self.degree == 1)

        edge_refined = {}
        new_verts = []

        def get_edge_vert(v1, v2):
            i = min(v1, v2)
            j = max(v1, v2)
            if i in edge_refined and j in edge_refined[i]:
                return edge_refined[i][j]
            else:
                if i not in edge_refined:
                    edge_refined[i] = {}
                new_vert = (self.vertices[i] + self.vertices[j])/2
                new_verts.append(new_vert)
                edge_refined[i][j] = self.num_vertices + len(new_verts) - 1
                return edge_refined[i][j]

        new_triangles = np.empty((self.num_triangles, 6), dtype=self.triangles.dtype)
        for i in range(self.num_triangles):
            p1, p2, p3 = self.triangles[i]
            p4 = get_edge_vert(p1, p3)
            p5 = get_edge_vert(p1, p2)
            p6 = get_edge_vert(p2, p3)
            new_triangles[i] = np.array([p1, p2, p3, p4, p5, p6])

        M = Mesh(np.concatenate((self.vertices, np.array(new_verts))), new_triangles, degree=2)
        M.degree_verts[1] = self.num_vertices

        return M

    def to_linear(self):
        """
        Converts a quadratic mesh to linear by replacing each quadratic triangle with
        4 linear ones.  Useful for plotting interfaces that only accept linear triangles,
        for example.
        """

        assert(self.degree == 2)

        new_tris = np.empty((self.num_triangles * 4, 3), dtype=self.triangles.dtype)
        for i in range(self.num_triangles):
            p1, p2, p3, p4, p5, p6 = self.triangles[i]
            new_tris[i*4 + 0] = np.array([p1, p5, p4])
            new_tris[i*4 + 1] = np.array([p5, p2, p6])
            new_tris[i*4 + 2] = np.array([p4, p5, p6])
            new_tris[i*4 + 3] = np.array([p4, p6, p3])

        return Mesh(self.vertices, new_tris, degree=1)


def get_phi_gradphi(xg, yg, wg, degree):
    if degree == 0:
        grad_phi = None
        phi = np.ones((len(xg), 1)) # k x 1
    elif degree == 1:
        grad_phi = np.broadcast_to(np.array([
            [-1., -1.],
            [1.0, 0.0],
            [0.0, 1.0]
        ])[:, :, None], (3, 2, len(xg))) # 3 x 2 x k
        phi = np.array([
            1 - xg - yg,
            xg,
            yg
        ]) # 3 x k
    elif degree == 2:
        grad_phi = np.empty((6, 2, len(xg))) # 6 x 2 x k
        for k, (x, y) in enumerate(zip(xg, yg)):
            grad_phi[:, :, k] = np.array([
                [-3 + 4 * x + 4 * y, -3 + 4 * y + 4 * x],
                [-1 + 4 * x, 0],
                [0, -1 + 4 * y],
                [-4 * y, 4 - 8 * y - 4 * x],
                [4 - 8 * x - 4 * y, -4 * x],
                [4 * y, 4 * x]
            ])
        phi = np.array([
            1 - 3*xg - 3*yg + 2*xg**2 + 2*yg**2 + 4*xg*yg,
            -xg + 2*xg**2,
            -yg + 2*yg**2,
            4*yg - 4*yg**2 - 4*xg*yg,
            4*xg - 4*xg**2 - 4*xg*yg,
            4*xg*yg
        ]) # 6 x k
    else:
        raise RuntimeError(f'Unimplemented degree: {degree}')
    return phi, grad_phi


def get_degree_tri(tri, deg):
    if deg == 0:
        return None
    elif deg == 1:
        return tri[:3]
    else:
        return tri


def integrate_element(tri, verts, fn, trial_degree, test_degree, integral_degree=7):
    xg, yg, wg = gausstri.deg[integral_degree].T

    # Linear mapping from global space to reference triangle
    x0, x1, x2 = verts[tri, 0][:3]
    y0, y1, y2 = verts[tri, 1][:3]
    J = np.array([
        [x1 - x0, y1 - y0],
        [x2 - x0, y2 - y0]
    ])
    J_det = la.det(J)

    # Transform cubature points into global space
    xg_global, yg_global = J @ np.row_stack((xg, yg)) + np.array([x0, y0])[:, None]

    # Get evaluations of phi and grad phi for given spaces
    u, grad_u = get_phi_gradphi(xg, yg, wg, trial_degree)
    grad_u = la.solve(J, grad_u)

    if trial_degree == test_degree:
        v, grad_v = u, grad_u
    else:
        v, grad_v = get_phi_gradphi(xg, yg, wg, test_degree)
        grad_v = la.solve(J, grad_v)

    u_tri = get_degree_tri(tri, trial_degree)
    v_tri = get_degree_tri(tri, test_degree)

    # Build up dictionary of variables to send to integration routine
    vals = {
        'xg': xg,
        'yg': yg,
        'wg': wg,
        'J': J,
        'xg_global': xg_global,
        'yg_global': yg_global,
        'u': u,
        'grad_u': grad_u,
        'u_tri': u_tri,
        'v': v,
        'grad_v': grad_v,
        'v_tru': v_tri
    }

    return J_det * fn(vals)


def assemble_operator(mesh, fn, element_degree=None, integral_degree=7):
    if element_degree is None:
        element_degree = (mesh.degree, mesh.degree)
    if not isinstance(element_degree, tuple):
        element_degree = (element_degree, element_degree)

    deg_trial, deg_test = element_degree

    # Sanity checks
    assert deg_trial <= mesh.degree and deg_test <= mesh.degree
    assert deg_trial in [0, 1, 2] and deg_test in [0, 1, 2]

    # Local element sizes will depend on degree of trial and test function spaces
    elem_sizes = {
        0: 1,
        1: 3,
        2: 6
    }
    N_rows = elem_sizes[deg_trial]
    N_cols = elem_sizes[deg_test]
    elem_numel = N_rows * N_cols

    # Collect local element matrices into COO format
    N_tri = mesh.num_triangles
    data = np.empty(elem_numel * N_tri)
    rows = np.empty(elem_numel * N_tri)
    cols = np.empty(elem_numel * N_tri)

    ii, jj = np.meshgrid(np.arange(N_rows), np.arange(N_cols), indexing='ij')
    ii = ii.flatten()
    jj = jj.flatten()

    # Integrate each element and accumulate into flat buffers
    for i in range(N_tri):
        data[i*elem_numel:(i+1)*elem_numel] = integrate_element(
            mesh.triangles[i], mesh.vertices, fn, deg_trial, deg_test, integral_degree).flatten()
        rows[i*elem_numel:(i+1)*elem_numel] = mesh.triangles[i][ii]
        cols[i*elem_numel:(i+1)*elem_numel] = mesh.triangles[i][jj]

    # "Build" the COO matrix, which will sum duplicate degrees of freedom
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(mesh.degree_verts[deg_trial], mesh.degree_verts[deg_test])).tocsr()
