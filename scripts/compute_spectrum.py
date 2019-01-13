"""Compute example spectrum."""
from __future__ import print_function, absolute_import
import numpy as np
from molsg.laplacemesh import compute_lb_fem

mol = np.load('data/ampc/npy/actives18.pqr0.50.41.2.npy')

eigs = compute_lb_fem(vertices=mol[0], faces=mol[1], k=100)

print('eigenvalues:', eigs.lam)
