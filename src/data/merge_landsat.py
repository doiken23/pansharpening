import argparse
from pathlib import Path
import tifffile
import numpy as np

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('base_name', type=str)
parser.add_argument('out', type=str)
args = parser.parse_args()

# read images
B_img = tifffile.imread(args.base_name + '_B2.TIF')
G_img = tifffile.imread(args.base_name + '_B3.TIF')
R_img = tifffile.imread(args.base_name + '_B4.TIF')

# merge and save images
img = np.stack((B_img, G_img, R_img), axis=0)
tifffile.imsave(str(Path(args.out).joinpath(args.base_name + '_RGB.tif')), img)
