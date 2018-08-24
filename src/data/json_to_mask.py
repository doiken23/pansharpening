import argparse
import json
from pathlib import Path

import tifffile
import numpy as np
from PIL import Image

from labelme import utils

# get argument
parser = argparse.ArgumentParser()
parser.add_argument('json', type=str)
parser.add_argument('img', type=str)
parser.add_argument('out', type=str)
args = parser.parse_args()

data = json.load(Path(args.json).open())
img = tifffile.imread(args.img)
if len(img.shape) == 3:
    img = img.transpose(1, 2, 0)

label_name_to_value = {'sea': 1}
lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

lbl = lbl.astype(np.uint8)
mask = np.ones(lbl.shape, dtype=np.uint8) - lbl
mask_img = Image.fromarray(mask)
mask_img.save(args.out, 'PNG', quality=100)
