# MolSG
Code for creating 3D Molecular Shape descriptors using Spectral and Diffusion Geometry. It is released under a 3-clause BSD license. See LICENSE.txt for details.

If you use MolSG in a scientific paper or derivative work, please cite the paper:
Alignment-Free Molecular Shape Comparison Using Spectral Geometry: The Framework
Matthew P. Seddon, David A. Cosgrove, Martin J. Packer, and Valerie J. Gillet
Journal of Chemical Information and Modeling Article ASAP
DOI: 10.1021/acs.jcim.8b00676

## Installation
1. `git clone` this directory
2. set MOLSG environment variable to the path where the module has been cloned. This tells the logger where to write the logs

## Descriptor computation
The descriptor has three stages:
1. Compute the spectrum of the Laplace-Beltrami operator over the molecular surface of the mesh
		 This is done using the Finite Element Method over a triangulated manifold mesh.

2. Compute a local descriptor for each vertex on the mesh using the spectrum
		 This descriptor describes the local variation of the shape using ideas from diffusion over surfaces

3. Compute a global descriptor for the shape.
		 This maps the local descriptor to a single descriptor in order to allow shape to shape comparison

## Work flows
Example work flows to show how to use the module are included in scripts. Scripts are run as python modules from the directory of the code. For example, `$ python -m scripts.process_meshes` converts the off meshes from the mesh generation outp ut to a directory of numpy versions. The mesh generation is carried out by TMSMesh from the pqr.tar.gz file which produced the off.tar.gz files in the `ampc` subdirectory.

To run the following examples you will need to extract the `data/ampc/off.tar.gz` file to `data/ampc/off`.

The example scripts are as follows:
- *process_meshes* : turn directory of off files to directory of numpy files.
- *compute_spectrum* : compute the spectrum of an example molecule in data/ampc
- *compute_WKS_descriptor* : computes the local geometry descriptor of an example molecule
- compute_covariance_descriptor : computes the global geometry covariance descriptor from a local geometry descriptor
- *compute_codebook* - takes a collection of WKS meshes and computes a codebook for a bag of features descriptor
- *compute_bagoffeatures_descriptor* : using a codebook that must be computed, this produces bag of features descriptor for a molecule and a plot of the histogram that is used as a descriptor
- *covariance_descriptor_screen* : computes the covariance descriptors for all example molecules in the data/ampc directory and returns the ranked list of actives
- *bagoffeatures_descriptor_screen* : computes the bagoffeatures descriptors for all example molecules in the data/ampc directory and returns the ranked list of actives
- *plot_surface_values* : plots the first eigenfunction of the LBO on the surface of an example molecule. NOTE: this requires mayavi to visualise the surface

## File types
The presumed output of a mesh generation program is `.off`. However, there is code in `molsg.utils` for `wrl` and the output from mgms. The first task is to turn the `off` files to `npy` format.

## Python modules
Note that the visualisation of the surfaces of the molecules requires Mayavi. As of September 2018, this can be installed in conda. However, the standard installation is Python 2.7. To install one Python 3.X the menpo channel is required.
The code can be run without it but the `scripts.plot_surface_values` script will require it.

## On Python versions
This code is designed to be compatible with Python 2.7 and 3.X. It has been tested with 2.7 and 3.6. However, Python's quirks means this may not be guaranteed going forward.
