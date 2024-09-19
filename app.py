import os, sys
import argparse
import numpy as np

from src.visualizer import CameraVisualizer
from src.loader import load_quick, load_nerf, load_colmap
from src.utils import load_image, rescale_cameras, recenter_cameras

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--format', default='quick', choices=['quick', 'nerf', 'colmap'])
parser.add_argument('--type', default=None, choices=[None, 'sph', 'xyz', 'elu', 'c2w', 'w2c'])
parser.add_argument('--no_images', action='store_true')
parser.add_argument('--mesh_path', type=str, default=None)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--scene_size', type=int, default=5)
parser.add_argument('--y_up', action='store_true')
parser.add_argument('--recenter', action='store_true')
parser.add_argument('--rescale', type=float, default=None)

args = parser.parse_args()

root_path = args.root

poses = []
legends = []
colors = []
images = None

if args.format == 'quick':
    poses, legends, colors, image_paths = load_quick(root_path, args.type)

elif args.format == 'nerf':
    poses, legends, colors, image_paths = load_nerf(root_path)

elif args.format == 'colmap':
    poses, legends, colors, image_paths = load_colmap(root_path)

if args.recenter:
    poses = recenter_cameras(poses)

if args.rescale is not None:
    poses = rescale_cameras(poses, args.rescale)

if args.y_up:
    for i in range(0, len(poses)):
        poses[i] = poses[i][[0, 2, 1, 3]]
        poses[i][1, :] *= -1
    
if not args.no_images:
    images = []
    for fpath in image_paths:
        if fpath is None:
            images.append(None)
            continue

        if not os.path.exists(fpath):
            images.append(None)
            print(f'Image not found at {fpath}')
            continue

        images.append(load_image(fpath, sz=args.image_size))

viz = CameraVisualizer(poses, legends, colors, images=images)
fig = viz.update_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True, show_ticklabels=True, show_background=True, y_up=args.y_up)

fig.show()
