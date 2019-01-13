"""Calculate and plot the 1st eigenfunction on the mesh."""
import sys
import numpy as np
from molsg import laplacemesh as lp
try:
    from mayavi import mlab
except ImportError:
    print('Mayavi module required')
    sys.exit(0)


def plot_mol(mesh, scalars=np.zeros(1), return_img=False):
    """Use mayavi to plot a molecule mesh."""
    vertices, faces = mesh
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
    mlab.triangular_mesh(x, y, z, faces, scalars=scalars)
    if return_img:
        return mlab.screenshot()
    else:
        mlab.show()


mol = np.load('data/ampc/npy/actives18.pqr0.50.41.2.npy')
eigs = lp.compute_lb_fem(vertices=mol[0], faces=mol[1], k=100)

plot_mol(mol, scalars=eigs.phi[:, 1])
