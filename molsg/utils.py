"""This program is a collection of utility functions for the data."""

from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import shutil
if sys.version_info.major == 2:
    from itertools import izip as zip


def wrl_to_np(wrl_name):
    """Return a mesh as a np data format of vertices and faces."""
    vertices = []
    faces = []
    colors = []
    in_geometry_obj = False
    scan_vertices = False
    scan_faces = False
    scan_colors = False
    with open(wrl_name) as wrl:
        for line in wrl:
            words = line.split()
            if "IndexedFaceSet" in words:
                in_geometry_obj = True
            elif 'point' in words:
                scan_vertices = True
            elif 'coordIndex' in words:
                scan_faces = True
            elif 'color' in words and '[' in words:
                scan_colors = True
            elif in_geometry_obj and scan_vertices:
                if len(words) == 3:
                    string_verts = [words[0], words[1], words[2][:-1]]
                    vertix = [float(i) for i in string_verts]
                    vertices.append(vertix)
                elif ']' in words:
                    scan_vertices = False
            elif in_geometry_obj and scan_faces:
                if len(words) == 4:
                    string_faces = [words[0], words[1], words[2]]
                    face = [int(i) for i in string_faces]
                    faces.append(face)
                elif ']' in words:
                    scan_faces = False
            elif in_geometry_obj and scan_colors:
                if len(words) == 3:
                    string_color = [words[0], words[1], words[2][:-1]]
                    color = [float(i) for i in string_color]
                    colors.append(color)
                elif ']' in words:
                    scan_colors = False
    # NOTE we ignore the normal values for now
    np_vertices = np.array(vertices)
    np_faces = np.array(faces)
    np_colors = np.array(colors)
    print("%d vertices and %d faces" % (len(np_vertices), len(np_faces)))
    return np_vertices, np_faces, np_colors


def get_shape_from_msms_output(name, norms=False):
    """Return a numpy representation of the surface from a mol name."""
    verts_file = name + '.vert'
    faces_file = name + '.face'
    verts = []
    vertsn = []
    faces = []
    with open(verts_file) as f:
        for line in f:
            words = line.split()
            if len(words) == 9:
                verts.append([float(x) for x in words[:3]])
                if norms:
                    vertsn.append([float(x) for x in words[3:6]])
    with open(faces_file) as f:
        for line in f:
            words = line.split()
            if len(words) == 5:
                faces.append([int(x) for x in words[:3]])
    np_vertices = np.array(verts)
    np_faces = np.array(faces) - 1  # correct for msms indexing
    clean_verts, clean_faces = clean_duplicates(np_vertices, np_faces)
    mesh = np.array([clean_verts, clean_faces])
    np.save(name, mesh)


def save_wrl_to_np_file(wrl_file):
    """Save np array of wrl mesh to a file."""
    mol_mesh = np.array(wrl_to_np(wrl_file))
    np.save(wrl_file, mol_mesh)


def clean_duplicates(verts, faces):
    """MSMS sometimes creates duplicate vertices so these need to be cleaned"""
    # set of coords
    # find duplicates
    coords = set([])
    dup = []
    for index, coordinate in enumerate(verts):
        coordinate_tup = tuple(coordinate)
        if coordinate_tup not in coords:
            coords.update([coordinate_tup])
        else:
            # if the coordinate is already there then perturb it
            dup.append(index)
            print(coordinate)
    print(dup)
    dup_verts = list(set(np.where(verts == verts[dup])[0]))
    faces_to_replace = []
    faces_to_delete = []
    faces_ix = []
    vert_to_keep = dup_verts[0]
    print(dup_verts)
    for index, face in enumerate(faces):
        num_dups = sum([1 if x in dup_verts else 0 for x in face])
        if num_dups == 1:
            faces_to_replace.append([vert_to_keep if x in
                                    dup_verts else x for x in face])
            faces_ix.append(index)
        elif num_dups > 1:
            # there will be two or 3 duplicated elements
            # which is a line or point
            # TODO ADD CHECK FOR LINE
            faces_to_delete.append(index)
    # Change faces
    for ix, face in zip(faces_ix, faces_to_replace):
        faces[ix] = face
    # Find new duplicates
    final_faces = []
    for ix, face in zip(faces_ix, faces_to_replace):
        if set(face) in [set(x) for x in final_faces]:
            faces_to_delete.append(ix)
    print(faces[faces_ix])
    # delete lines and points
    print('pre delete', len(faces))
    faces = np.delete(faces, faces_to_delete, axis=0)
    print('post delete', len(faces))
    # NOTE this leaves some retundant vertices in the array that are never used
    # but this shouldn't be too much of a problem
    return verts, faces


def get_shape_from_off(fname, debug=False):
    """Save numpy pickle of mesh"""
    vertices = []
    faces = []
    in_verts = False
    in_faces = False

    with open(fname) as f:
        for line in f:
            words = line.split()
            if not in_verts and len(words) == 3:
                in_verts = True
            elif in_verts and len(words) == 3:
                vertix = [float(n) for n in words]
                vertices.append(vertix)
            elif in_verts and len(words) == 4:
                in_verts = False
                face = [int(n) for n in words]
                faces.append(face[1:])
                in_faces = True
            elif in_faces and len(words) == 4:
                face = [int(n) for n in words]
                faces.append(face[1:])

    np_vertices = np.array(vertices)
    np_faces = np.array(faces)
    if debug:
        print("%d vertices and %d faces" % (len(np_vertices), len(np_faces)))
    mesh = np.array([np_vertices, np_faces])
    return mesh


def get_shapes_from_dir(dname):
    offdir = '%s/off/' % dname
    npydir = '%s/npy/' % dname
    if not os.path.exists(npydir):
        os.makedirs(npydir)
    for fname in os.listdir(offdir):
        mesh = get_shape_from_off('%s/%s' % (offdir, fname))
        np.save('%s/%s' % (npydir, fname[:-4]), mesh)


def get_mesh_from_TMS(target, target_dir, ligands=None, save_to=None):
    """Get a TMS mesh from a pqr file.

    ligands are indices of ligands in file list

    Use target as a local file, meaning that you can use pqr file from any
    target location.
    """
    target_dir = target_dir + target

    # get ligands to generate meshes
    ligs = np.array(os.listdir('%s/pqr' % target_dir))
    if not ligands:
        ligs = ligs[ligands]

    # move ligands to tms and generate meshes
    tms = os.environ['HOME'] + '/TMSmesh_2012_11'
    os.system('rm %s/pqr/*' % tms)
    for l in ligs:
        shutil.copy('%s/pqr/%s/%s_1.pqr' % (target_dir, l, l), '%s/pqr/' % tms)
    os.system('cd %s; bash job.sh; cd -' % tms)
    print('generated meshes')
    destination = '../data/test_sets/xtarget/%s/' % target
    if not os.path.isdir(destination):
        os.makedirs('%s/off/' % destination)
        os.makedirs('%s/npy/' % destination)
    os.system('mv %s/off/* %s/off/' % (tms, destination))
    print('copied to local dir')
    print(len(os.listdir('%s/off/' % destination)))
    for off in os.listdir('%s/off' % destination):
        mesh = get_shape_from_off('%s/off/%s' % (destination, off))
        np.save('%s/npy/%s.npy' % (destination, off), mesh)


def get_meshes_from_sdf_dir(sdf_dir, target_dir, ligands=None, save_to=None):
    """Get a TMS mesh from a pqr file.

    ligands are indices of ligands in file list

    Use target as a local file, meaning that you can use pqr file from any
    target location.
    """
    target_dir = target_dir + sdf_dir

    # get ligands to generate meshes
    ligs = np.array(os.listdir('%s/pqr' % target_dir))
    if ligands:
        ligs = ligs[ligands]

    # move ligands to tms and generate meshes
    tms = os.environ['HOME'] + '/TMSmesh'
    os.system('rm %s/pqr/*' % tms)
    for l in ligs:
        shutil.copy('%s/pqr/%s' % (target_dir, l), '%s/pqr/' % tms)
    os.system('cd %s; bash job.sh; cd -' % tms)
    print('generated meshes')
    if not os.path.isdir('%s/off/' % target_dir):
        os.makedirs('%s/off/' % target_dir)
        os.makedirs('%s/npy/' % target_dir)
    os.system('mv %s/off/* %s/off/' % (tms, target_dir))
    print('copied to local dir')
    print(len(os.listdir('%s/off/' % target_dir)))
    for off in os.listdir('%s/off' % target_dir):
        mesh = get_shape_from_off('%s/off/%s' % (target_dir, off))
        np.save('%s/npy/%s.npy' % (target_dir, off), mesh)
