"""Compute the local geometry descriptors of a molecule."""
from __future__ import division, print_function, absolute_import
import logging
import os

import numpy as np
from collections import namedtuple

# Set up logging
log = logging.getLogger('LocalGeometry')
log.setLevel(logging.DEBUG)
logfile = logging.FileHandler('{}/logs/LocalGeometry.log'
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


LGDparams = namedtuple('LGDparams', 'function parameters')


class LocalGeometryError(Exception):
    """Exception class for LocalGeometry."""

    pass


class LocalGeometry(object):
    """Laplacian Mesh object to represent a molecule.

       This class can contain the mesh, eigenvalues, and the local geometry
       descriptor for a molecule. However, for large sets of molecules, it
       may be preferable to use the helper functions to save memoryself."""

    def __init__(self, eigs, params):
        """Initialize the parameters of the mesh.

        Args
        eigs: namedtuple containing information on
              vertices and faces of a mesh

        params: namedtuple with the LGDparams.function being the functional
                form and LGDparams.parameters being the corresponding
                parameters
        """
        self.eigs = eigs
        if params.function == 'HKS':
            self.descriptor = HKS_descriptor(eigs, time=params.parameters)
        elif params.function == 'WKS':
            self.descriptor = WKS_descriptor(eigs, evals=params.parameters)
        else:
            raise LocalGeometryError('{} not a valid functional form'
                                     .format(params.function))


"""
methods
"""


def heat_kernel_matrix(eigensystem, t=20, sample=np.array([])):
    """Compuate heat kernel matrix for all vertices given eigensystem."""
    lam, phi = eigensystem
    if len(sample) > 0:
        phi = phi[sample]
    return np.einsum('ik, jk -> ij', np.exp(- abs(lam) * t) * phi, phi)


def WKS_descriptor(eigensystem, evals=100, variance=7, sample=None):
    """Compute wave kernel signature for eigen decomposition of LB.

    Code implemented from MATLAB example on M. Aubry's website:
    http://imagine.enpc.fr/~aubrym/projects/wks/index.html

    Args:
    evals: Number of WKS evaluations

    variance: Variance of the Gaussian function. This is set to
              7 * delta, where delta is the increment of the
              log eigenscale.

    """
    lam, phi = eigensystem
    if sample:
        phi = phi[sample]
    WKS = np.zeros((phi.shape[0], evals))
    # WKS looks at energies rather than times
    log_E = np.log(np.maximum(np.abs(lam), 1e-6))  # a 1 x num_eigs vector
    e = np.linspace(log_E[1], max(log_E) / 1.02, evals)
    sigma = (e[1] - e[0]) * variance
    phi_squared = np.multiply(phi, phi)
    E = np.tile(e, (lam.shape[0], 1)).T
    Tau = np.exp(-(E - log_E)**2 / (2 * sigma**2))
    WKS = np.tensordot(Tau, phi_squared, (1, 1)).T
    C = (np.exp(-(np.tile(e, (lam.shape[0], 1)) -
                  np.tile(log_E, (evals, 1)).T) ** 2 / (2 * sigma ** 2))
         ).sum(axis=0)
    descriptor = WKS / C
    return descriptor


def HKS_descriptor(eigensystem, sample=None,
                   time=np.arange(start=1, stop=100, step=5)):
    """Compute hest kernel signature from eigensystem solution.

    The HKS descriptor is a vector field defined over the manifold. The
    vector is defined as the scalar heat values for a given point in time.

    Args:
    time : time range over which to evaluate the descriptor
    """
    lam, phi = eigensystem
    if sample:
        phi = phi[sample]
    return np.dot(phi * phi, np.exp(np.outer(time, -abs(lam))).T)


def heat_kernel(eigs, x, y, t):
    """Compute heat kernel between two points at a time t."""
    lam, phi = eigs
    phi_x = phi[x]
    phi_y = phi[y]
    hk = np.exp(- abs(lam) * t) * phi_x * phi_y
    return hk.sum()


def diffusion_distance(eigs, x, y, t):
    """Compute the diffusion distance between two vertices.

    Args:
    eigs: A tuple of eigenvalues and eigenvectors
    x: The index of a vertex
    y: The index of a vertex

    Note: the sqaured distance for any two points at time t using the heat
    kernel is:

    $$d^2(x,y) = h_t(x,x) + h_t(y,y) - 2h_t(x,y)$$

    """
    return np.sqrt(heat_kernel(eigs, x, x, t) +
                   heat_kernel(eigs, y, y, t) -
                   2 * heat_kernel(eigs, x, y, t))
