"""Compute example WKS compute_WKS_descriptor."""
from __future__ import print_function, absolute_import
import os
import numpy as np
from sklearn.preprocessing import normalize
from molsg.laplacemesh import compute_lb_fem
from molsg.localgeometry import WKS_descriptor
from molsg.bagoffeatures import compute_codebook
evals = 100

data_dir = 'data/ampc/npy/'
mols = (np.load('{}/{}'.format(data_dir, m)) for m in os.listdir(data_dir))
eigs = (compute_lb_fem(vertices=mol[0], faces=mol[1], k=100) for mol in mols)
wks = [WKS_descriptor(eig, evals=evals) for eig in eigs]
descriptor_space = np.concatenate(wks)
descriptor_space = normalize(descriptor_space)

codebook = compute_codebook(descriptor_space, num_codewords=50)

print('codebook shape:', codebook.shape)

if not os.path.exists('data/codebooks/example.npy'):
    os.makedirs('data/codebooks/')
    np.save('data/codebooks/example.npy', codebook)
    print('saved codebook to data/codebooks')
