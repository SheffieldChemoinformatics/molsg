"""Prepare mesh data for a target given a directory of off format meshes.

This assumes that the data are in the following format:
target/{off,npy}

The required input is the path to the target directory.
"""
from __future__ import absolute_import
from molsg.utils import get_shapes_from_dir

target_path = 'data/ampc'
print('Getting npy meshes from ', target_path)

get_shapes_from_dir(target_path)
