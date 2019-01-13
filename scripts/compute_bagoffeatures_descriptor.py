"""Compute example WKS Bag of Features descriptor."""
from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from molsg.laplacemesh import compute_lb_fem
from molsg.localgeometry import WKS_descriptor
import molsg.bagoffeatures as bf


evals = 100
num_codewords = 50

if not os.path.exists('data/codebooks/example.npy'):
    print('Must compute codebook first.')
    sys.exit(0)

codebook = np.load('data/codebooks/example.npy')

mol = np.load('data/ampc/npy/actives18.pqr0.50.41.2.npy')
eigs = compute_lb_fem(vertices=mol[0], faces=mol[1], k=100)
wks = WKS_descriptor(eigs, evals=evals)
wks = normalize(wks)
bof = bf.hq_descriptor(wks, codebook)

plt.plot(bof)
plt.title('Bag of features descriptor for actives 18')
plt.show()
