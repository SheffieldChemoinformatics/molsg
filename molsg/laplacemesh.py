"""Program to generate laplacian mesh of a molecule."""
from __future__ import division, print_function, absolute_import
import logging
import os
from collections import namedtuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Set up logging
log = logging.getLogger('LaplacianMesh')
log.setLevel(logging.DEBUG)
logfile = logging.FileHandler('{}/logs/LaplacianMesh.log'
                              .format(os.environ['MOLSG']),
                              mode='a')
logfile.setLevel(logging.DEBUG)
logconsole = logging.StreamHandler()
logconsole.setLevel(logging.ERROR)
logformat = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logfile.setFormatter(logformat)
logconsole.setFormatter(logformat)
log.addHandler(logfile)
log.addHandler(logconsole)

# Define mesh and eigenvalue objects
Mesh = namedtuple('Mesh', 'vertices faces')
Eigs = namedtuple('Eigs', 'lam phi')


class LaplacianMeshError(Exception):
    """Exception class for LaplacianMesh."""

    pass


class LaplacianMesh(object):
    """Laplacian Mesh object to represent a molecule.

       This class can contain the mesh, eigenvalues, and the local geometry
       descriptor for a molecule. However, for large sets of molecules, it
       may be preferable to use the helper functions to save memory."""

    def __init__(self, mesh, k=400, name=None):
        """Initialize the parameters of the mesh.

        Args:
        mesh: namedtuple containing information on
              vertices and faces of a mesh

        k: number of eigenvalues to take

        name: if True then take the name of the ligand

        Note : the Name will be useful for classification purposes.
        """
        self.mesh = mesh
        self.num_vertices = len(self.mesh.vertices)
        self.num_faces = len(self.mesh.faces)
        self.eigs = np.array([])
        self.name = name
        self.k = k

    def compute_lb_fem(self, save=False):
        """
        Compute the spectrum of the Laplace-Beltrami operator using the finite
        element method.

        Note ::
            All points must be on faces. Otherwise, a singular matrix error
            is generated when inverting D.
        Args
        points : list of lists of 3 floats
                 x,y,z coordinates for each vertex
        faces : list of lists of 3 integers each list contains indices
                to vertices that form a triangle on a mesh
        """
        self.eigs = compute_lb_fem(vertices=self.mesh.vertices,
                                   faces=self.mesh.faces,
                                   k=self.k)
        self.has_eigs = True
        if save:
            np.save(save, self.eigs)


"""
methods
"""


def compute_lb_fem(vertices, faces, k):
    """
    Compute the Laplace-Beltrami operator using the finite element method.

    Note ::
        All points must be on faces. Otherwise, a singular matrix error
        is generated when inverting D.

        Code implemented from MATLAB tutorial [http://reuter.mit.edu/tutorial/].
        However, that link is no longer live, for ShapeDNA using FEM see
        http://reuter.mit.edu/software/shapedna/
    Args
    points : list of lists of 3 floats
             x,y,z coordinates for each vertex
    faces : list of lists of 3 integers each list contains indices to
            vertices that form a triangle on a mesh
    """
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    if num_vertices <= k:
        raise LaplacianMeshError('Not enough vertices ({} <= {}).'
                                 .format(num_vertices, k))
    # Construct local matrices
    tB = (np.ones((3, 3)) + np.eye(3)) / 24.0

    tA00 = np.array([[0.5, -0.5, 0.0],
                     [-0.5, 0.5, 0.0],
                     [0.0, 0.0, 0.0]])

    tA11 = np.array([[0.5, 0.0, -0.5],
                     [0.0, 0.0, 0.0],
                     [-0.5, 0.0, 0.5]])

    tA0110 = np.array([[1.0, -0.5, -0.5],
                       [-0.5, 0.0, 0.5],
                       [-0.5, 0.5, 0.0]])

    # Store one for each face, this enables us to vectorize everything
    # Note: we can treat them as (num_faces, 3, 3) tensors for efficient
    # computation using einsum
    tB = np.tile(tB, (num_faces, 1, 1))
    tA00 = np.tile(tA00, (num_faces, 1, 1))
    tA11 = np.tile(tA11, (num_faces, 1, 1))
    tA0110 = np.tile(tA0110, (num_faces, 1, 1))

    # difference vectors from v1 --> v2 and v1 --> v3 for each face
    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]
    v2v1 = v2 - v1
    v3v1 = v3 - v1

    # Calculate the squared lengths of the difference vector and their dot
    # product then map to each face using einsum
    a0 = np.einsum('ij, ij, kl -> ikl', v3v1, v3v1, np.ones((3, 3)))
    a1 = np.einsum('ij, ij, kl -> ikl', v2v1, v2v1, np.ones((3, 3)))
    a0110 = np.einsum('ij, ij, kl -> ikl', v2v1, v3v1, np.ones((3, 3)))

    # Compute volume for each triangle using cross product and remove
    # divide by zero errors by replacing zero matrices with the mean
    cross_products = np.cross(v2v1, v3v1)
    volumes = np.sqrt(np.einsum('ij, ij, kl -> ikl',
                                cross_products,
                                cross_products,
                                np.ones((3, 3))))
    volumes[volumes == 0] = np.mean(volumes)

    # Construct local A and B matrices for each face
    localB = volumes * tB
    localA = (1.0 / volumes) * (a0 * tA00 + a1 * tA11 - a0110 * tA0110)

    # Construct sparse matrix, flatten to ensure sparsity
    # Cast as np.int to avoid errors in Python 3
    J_indices = np.einsum('ij,jk -> ijk',
                          faces,
                          np.ones((3, 3))).flatten().astype(np.int)
    I_indices = np.einsum('ij,jk -> ikj',
                          faces,
                          np.ones((3, 3))).flatten().astype(np.int)
    localA = localA.flatten()
    localB = localB.flatten()
    A = sparse.csr_matrix((localA, (I_indices, J_indices)))
    B = sparse.csr_matrix((localB, (I_indices, J_indices)))

    # Use ARPACK to solve sparse system using sigma=0 inversion
    try:
        eigs = eigsh(A, k=k, M=B, sigma=0, which='LM')
        return Eigs(eigs[0], eigs[1])
    except RuntimeError as error:
        # eigsh is based on Fortran's ARPACK. For some reason, a
        # RuntimeError occasionally occurs when the sparse matrix A cannot
        # be inverted. Nevertheless, the dense matrix can be. There are
        # two options: run eigsh on the dense matrices (possible slow) or
        # increase the sigma value incrementally to 0.001, less precise.
        # For now, the approach taken is the dense solver. If this becomes
        # too slow then we will have to consider using a new sigma or
        # a preconditioned partial approach.
        # http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        log.error('sparse calculation failed: {error}'.format(error=error))
        log.info('trying dense approach...')
        eigs = eigsh(A.A, k=k, M=B.A, sigma=0, which='LM')
        return Eigs(eigs[0], eigs[1])
