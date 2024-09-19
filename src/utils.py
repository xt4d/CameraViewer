import os
import numpy as np
from PIL import Image


def load_image(fpath, sz=256):
    img = Image.open(fpath)
    img = img.resize((sz, sz))
    return np.asarray(img)[:, :, :3]


def spherical_to_cartesian(sph):

    theta, azimuth, radius = sph

    return np.array([
        radius * np.sin(theta) * np.cos(azimuth),
        radius * np.sin(theta) * np.sin(azimuth),
        radius * np.cos(theta),
    ])


def cartesian_to_spherical(xyz):

    xy = xyz[0]**2 + xyz[1]**2
    radius = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    azimuth = np.arctan2(xyz[1], xyz[0])

    return np.array([theta, azimuth, radius])


def elu_to_c2w(eye, lookat, up):

    if isinstance(eye, list):
        eye = np.array(eye)
    if isinstance(lookat, list):
        lookat = np.array(lookat)
    if isinstance(up, list):
        up = np.array(up)

    l = eye - lookat
    if np.linalg.norm(l) < 1e-8:
        l[-1] = 1
    l = l / np.linalg.norm(l)

    s = np.cross(l, up)
    if np.linalg.norm(s) < 1e-8:
        s[0] = 1
    s = s / np.linalg.norm(s)
    uu = np.cross(s, l)

    rot = np.eye(3)
    rot[0, :] = -s
    rot[1, :] = uu
    rot[2, :] = l
    
    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = eye

    return c2w


def c2w_to_elu(c2w):

    w2c = np.linalg.inv(c2w)
    eye = c2w[:3, 3]
    lookat_dir = -w2c[2, :3]
    lookat = eye + lookat_dir
    up = w2c[1, :3]

    return eye, lookat, up


def qvec_to_rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])


def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def recenter_cameras(c2ws):

    is_list = False
    if isinstance(c2ws, list):
        is_list = True
        c2ws = np.stack(c2ws)
  
    center = c2ws[..., :3, -1].mean(axis=0)
    c2ws[..., :3, -1] = c2ws[..., :3, -1] - center

    if is_list:
         c2ws = [ c2w for c2w in c2ws ]

    return c2ws


def rescale_cameras(c2ws, scale):

    is_list = False
    if isinstance(c2ws, list):
        is_list = True
        c2ws = np.stack(c2ws)
  
    c2ws[..., :3, -1] *= scale

    if is_list:
         c2ws = [ c2w for c2w in c2ws ]

    return c2ws
