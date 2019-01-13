"""Program to produce covariance descriptors for a given local descriptor."""
from __future__ import print_function, absolute_import, division

import copy

import numpy as np

import molsg.laplacemesh as lp


class CovarianceDescriptorError(Exception):
    """Exception class for LaplacianMesh."""

    pass


def patch_covariance(patch):
    """Compute covaraince descriptor of an input patch."""
    means = patch.mean(axis=0)
    patch_bar = patch - means
    return np.dot(patch_bar.T, patch_bar)


def get_ring(mesh):
    """Compute the one-ring for a mesh.

    Args
    mesh:: A laplacemesh.Mesh object
    Returns
    ring_dict:: a dictionary of one-rings per vertex."""

    ring_dict = {}
    for face in mesh.faces:
        for vert in face:
            try:
                ring_dict[vert] |= set(face)
            except KeyError:
                ring_dict[vert] = set(face)
    return ring_dict


def get_next_ring(ring_dict):
    new_dict = {}
    for vert in ring_dict:
        new_dict[vert] = copy.deepcopy(ring_dict[vert])
        for neighbour in ring_dict[vert]:
            new_dict[vert] |= ring_dict[neighbour]
    return new_dict


def covariance_descriptor(wks, mesh=None, ring_level=0):
    if ring_level == 0:
        return patch_covariance(wks)
    if not isinstance(mesh, lp.Mesh):
        raise CovarianceDescriptorError('mesh is not a laplacemesh.Mesh object')
    N, D = wks.shape
    C = np.zeros((D, D))
    rings = get_ring(mesh)
    while ring_level > 1:
        rings = get_next_ring(rings)
        ring_level -= 1
    for vert in rings:
        ring = list(rings[vert])
        patch = wks[ring]
        C += patch_covariance(patch)
    return C / N
