import os, sys
import argparse
import numpy as np

from src.visualizer import CameraVisualizer
from src.loader import load_quick, load_nerf, load_colmap
from src.utils import load_image

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--format', default='quick', choices=['quick', 'nerf', 'colmap'])
parser.add_argument('--type', default=None, choices=[None, 'sph', 'xyz', 'elu', 'c2w', 'w2c'])
parser.add_argument('--no_images', action='store_true')
parser.add_argument('--mesh_path', type=str, default=None)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--scene_size', type=int, default=5)

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
fig = viz.update_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True, show_ticklabels=True, show_background=True)

fig.show()