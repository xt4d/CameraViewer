import os
import numpy as np
import json

from .utils import elu_to_c2w, spherical_to_cartesian, load_image, qvec2rotmat, rotmat


def load_quick(root_path, type):

    poses = []
    legends = []
    colors = []
    image_paths = []

    pose_root = os.path.join(root_path, 'poses')
    print(f'Load poses from {pose_root}')

    image_root = os.path.join(root_path, 'images')
    print(f'Load images from {image_root}')

    fname_list = os.listdir(pose_root)

    for fname in fname_list:

        vals = fname.split('.')
        fid, ext = vals[0], vals[-1]

        fpath = os.path.join(pose_root, fname)

        if ext == 'npy':
            mat = np.load(fpath)
        elif ext == 'txt':
            mat = np.loadtxt(fpath)

        if type == 'c2w':
            c2w = mat
            if c2w.shape[0] == 3:
                c2w = np.concatenate([c2w, np.zeros((1, 4))], axis=0)
                c2w[-1, -1] = 1

        if type == 'w2c':
            w2c = mat
            if w2c.shape[0] == 3:
                w2c = np.concatenate([w2c, np.zeros((1, 4))], axis=0)
                w2c[-1, -1] = 1
            c2w = np.linalg.inv(w2c)

        elif type == 'elu':
            eye = mat[0, :]
            lookat = mat[1, :]
            up = mat[2, :]
            c2w = elu_to_c2w(eye, lookat, up)

        elif type == 'sph' or type == 'xyz':

            assert (mat.size == 3)
    
            if type == 'sph':
                eye = spherical_to_cartesian((np.deg2rad(mat[0]), np.deg2rad(mat[1]), mat[2]))
            else:
                eye = mat

            lookat = np.zeros(3)
            up = np.array([0, 0, 1])
            c2w = elu_to_c2w(eye, lookat, up)

        poses.append(c2w)
        legends.append(fid)
        colors.append('blue')

        img_paths = [ os.path.join(image_root, f'{fid}.{ext}') for ext in ['png', 'jpg', 'jpeg']]
        img_paths = [ fpath for fpath in img_paths if os.path.exists(fpath) ]
        if len(img_paths) < 1:
            image_paths.append(None)
        else:
            image_paths.append(img_paths[0])

    return poses, legends, colors, image_paths


def load_nerf(root_path):

    poses = []
    legends = []
    colors = []
    image_paths = []

    pose_path = os.path.join(root_path, 'transforms.json')
    print(f'Load poses from {pose_path}')

    with open(pose_path, 'r') as fin:
        jdata = json.load(fin)

    for fi, frm in enumerate(jdata['frames']):

        c2w = np.array(frm['transform_matrix'])
        poses.append(c2w)
        colors.append('blue')

        if 'file_path' in frm:
            fpath = frm['file_path']
            fname = os.path.basename(fpath)
            
            legends.append(fname)
            image_paths.append(os.path.join(root_path, fpath))
        else:
            legends.append(str(fi))
            images.append(None)


    return poses, legends, colors, image_paths


def load_colmap(root_path):

    poses = []
    legends = []
    colors = []
    image_paths = []

    pose_path = os.path.join(root_path, 'images.txt')
    print(f'Load poses from {pose_path}')
    
    fin = open(pose_path, 'r')

    up = np.zeros(3)

    i = 0
    for line in fin:
        line = line.strip()
        if line[0] == "#":
            continue
        i = i + 1
        if  i % 2 == 0:
            continue
        elems = line.split(' ')

        fname = '_'.join(elems[9:])
        legends.append(fname)

        fpath = os.path.join(root_path, 'images', fname)
        image_paths.append(fpath)

        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        rot = qvec2rotmat(-qvec)
        tvec = tvec.reshape(3)

        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, -1] = tvec
        c2w = np.linalg.inv(w2c)

        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 # flip whole world upside down

        up += c2w[0:3,1]

        poses.append(c2w)
        colors.append('blue')

    fin.close()

    up = up / np.linalg.norm(up)
    up_rot = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    up_rot = np.pad(up_rot,[0,1])
    up_rot[-1, -1] = 1

    for i in range(0, len(poses)):
        poses[i] = np.matmul(up_rot, poses[i])

    return poses, legends, colors, image_paths