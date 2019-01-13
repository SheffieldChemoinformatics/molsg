"""Example virtual screen of Bag of Features descriptors."""
from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance

from molsg.laplacemesh import compute_lb_fem
from molsg.localgeometry import WKS_descriptor
import molsg.bagoffeatures as bf


evals = 100
num_codewords = 50

if not os.path.exists('data/codebooks/example.npy'):
    print('Must compute codebook first.')
    sys.exit(0)

codebook = np.load('data/codebooks/example.npy')

mol_names = sorted(os.listdir('data/ampc/npy/'))
mols = (np.load('data/ampc/npy/{}'.format(m)) for m in mol_names)
eigs = (compute_lb_fem(vertices=m[0], faces=m[1], k=20) for m in mols)
wks = (normalize(WKS_descriptor(e, evals=evals)) for e in eigs)
bofs = np.asarray([bf.hq_descriptor(w, codebook) for w in wks])

# use first molecule as reference
similarity = distance.cdist(bofs[:10], bofs, metric='cosine')
labels = np.asarray([1 if m[0] == 'a' else 0 for m in mol_names])

rankings = np.argsort(similarity)

print('ranking:', labels[rankings])
