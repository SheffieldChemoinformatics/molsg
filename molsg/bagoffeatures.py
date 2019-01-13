"""Program to implement the bag of features similarity search for HKS."""

from __future__ import print_function, division
import os

import numpy as np
import scipy.spatial.distance as scidist
from scipy.cluster import vq
from scipy.spatial import distance
from sklearn import cluster
import sklearn.metrics.pairwise as pairwise
from sklearn.preprocessing import normalize

import logging
log = logging.getLogger('BagOfFeatures')
log.setLevel(logging.DEBUG)
logfile = logging.FileHandler('{}/logs/BagOfFeatures.log'
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


class BagOfFeaturesError(Exception):
    """Bag of Features exception."""

    pass


class BagOfFeatures(object):
    """Bag of Features Class.

    The Bag of Features class first searches feature space for k clusters
    to get obtain a basic shape vocabulary. Then histograms are created to
    get a probability distribution for a point belonging to a feature. Finally
    a matrix is constructed to give some shape expressions.

    References:

    Bronstein, A. M., Bronstein, M. M., Guibas, L. J., & Ovsjanikov, M.
    (2011). Shape Google: Geometric Words and Expressions for Invariant
    Shape Retrieval. ACM Trans. Graph., 30(1), 1:1-1:20.

    Ovsjanikov, M., Bronstein, A. M., Bronstein, M. M., & Guibas, L. J.
    (2009). Shape Google: a computer vision approach to isometry invariant
    shape retrieval (pp. 320-327). IEEE.

    """

    def __init__(self,
                 codebook,
                 encoding='HQ'):
        """
        Initialize BagOfFeatures with the parameters for the model.

        Args:
        encoding: Type of encoding for the histogram: HQ, SA, KNN
        codebook: codebook for histogram encoding
        """
        self.encoding = encoding
        self.codebook = codebook

    def transform_single(self, local_geometry_descriptor):
        """Transform a single local geometry descriptor into a histogram."""
        if self.encoding == 'HQ':
            return hq_descriptor(local_geometry_descriptor, self.codebook)
        elif self.encoding == 'SA':
            return sa_histogram(local_geometry_descriptor, self.codebook)
        elif self.encoding == 'KNN':
            return knn_histogram(local_geometry_descriptor, self.codebook)
        else:
            log.error('Encoding must be ether HQ, SA, or KNN')
            raise BagOfFeaturesError('Encoding must be ether HQ, SA, or KNN')

    def transform(self, local_geometry_descriptors):
        """Transform list of local geometry descriptors."""
        if self.encoding == 'HQ':
            return np.array([hq_descriptor(lgd, self.codebook)
                             for lgd in local_geometry_descriptors])
        elif self.encoding == 'SA':
            return np.array([sa_histogram(lgd, self.codebook)
                             for lgd in local_geometry_descriptors])
        elif self.encoding == 'KNN':
            return np.array([knn_histogram(lgd, self.codebook)
                             for lgd in local_geometry_descriptors])
        else:
            log.error('Encoding must be ether HQ, SA, or KNN, not {}'
                      .format(self.encoding))
            raise BagOfFeaturesError('Encoding must be ether HQ, SA, or KNN, not {}'
                                     .format(self.encoding))


"""
methods
"""


def calculate_sigma(codebook):
    """Calculate softmax sigma for a given codebook.

    The sigma is calculated as the square root of double the median distance in
    the codebook.
    """
    distances = distance.pdist(codebook, 'euclidean')
    variance = 2 * np.median(distances)
    return np.sqrt(variance)


def compute_codebook(descriptor_space, num_codewords):
    """Compute codebook for bag of features descriptor using k-means.

    The descriptor space is a collection of local geometry descriptors. This
    needs to be in the format of a matrix, rather than a list or a tensor.
    """
    if isinstance(descriptor_space, list):
        descriptor = np.array(descriptor_space)
    if len(descriptor_space.shape) > 2:
        log.error('descriptor_space is a tensor but must be 2-dim array')
        raise BagOfFeaturesError('descriptor_space must be 2-dim array')
    return (cluster.MiniBatchKMeans(n_clusters=num_codewords)
                   .fit(descriptor_space)
                   .cluster_centers_)


def hq_descriptor(molecule, codebook,
                  norm='l1', return_encoding=False):
    """Compute the HQ endoding descriptor."""
    labels, _ = vq.vq(molecule, codebook)
    if return_encoding:
        return labels
    frequency = np.array([len(np.where(labels == i)[0])
                          for i in range(codebook.shape[0])],
                         dtype=np.float64)
    return normalize(frequency.reshape(1, -1), norm=norm)[0]


def sa_histogram(mol_descriptor, codebook,
                 sigma=None,
                 normalized=True,
                 norm='l1',
                 return_encoding=False):
    """Compute the histogram for a given descriptor and the codebook."""
    if not sigma:
        sigma = calculate_sigma(codebook=codebook)
    num_vertices = mol_descriptor.shape[0]
    if normalized:
        mol_descriptor = normalize(mol_descriptor, norm='l2')

    squared_L2_distance = scidist.cdist(codebook,
                                        mol_descriptor,
                                        metric='sqeuclidean').T
    softmax_distance = np.exp(-0.5 * squared_L2_distance / sigma**2)
    if return_encoding:
        return softmax_distance
    mol_histogram = normalize(softmax_distance, norm=norm).sum(axis=0)
    return mol_histogram / num_vertices


def knn_histogram(molecule, codebook,
                  knn=3,
                  norm='l1',
                  sigma=None,
                  dist='softmax',
                  return_encoding=False):
    squared_L2_distance = distance.cdist(codebook, molecule,
                                         metric='sqeuclidean').T
    softmax_distance = np.exp(-0.5 * squared_L2_distance / sigma**2)
    if dist == 'softmax':
        n_closest_features = np.argsort(softmax_distance, axis=1)[:, -knn:]
    elif dist == 'sqeuclidean':
        n_closest_features = np.argsort(squared_L2_distance, axis=1)[:, :knn]
    else:
        raise ValueError("distance not recognised")
    nearest_features = np.zeros(softmax_distance.shape)
    for row, ix in zip(nearest_features, n_closest_features):
        row[ix] = 1
    if dist == 'softmax':
        X = softmax_distance * nearest_features
        X = normalize(X, norm='l1')
    elif dist == 'sqeuclidean':
        X = nearest_features
    if return_encoding:
        return X
    return normalize(X.sum(axis=0).reshape(1, -1), norm=norm)[0]


def histogram_similarity(hist1, hist2):
    """Calculate the similarity between two molecule histograms.

    Args:
    hist1: molecule histogram

    hist2: molecule histogram
    """
    return pairwise.chi2_kernel(hist1, hist2)
