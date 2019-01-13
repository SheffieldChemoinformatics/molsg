"""Compute example WKS compute_WKS_descriptor."""
from __future__ import print_function, absolute_import
import numpy as np
from molsg.laplacemesh import compute_lb_fem
from molsg.localgeometry import WKS_descriptor

evals = 100

mol = np.load('data/ampc/npy/actives18.pqr0.50.41.2.npy')
eigs = compute_lb_fem(vertices=mol[0], faces=mol[1], k=100)
wks = WKS_descriptor(eigs, evals=evals)

print('number of vertices:', mol[0].shape[0])
print('evals:', evals)
print('WKS descriptor shape:', wks.shape)
