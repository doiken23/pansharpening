import time
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('size', type=int)
    args = parser.parse_args()
    return args

def make_rgb(args, img_name, img_dir):
    # merge rgb bands
    rgb = []
    for i in range(4, 1, -1):
        img_path = img_dir.joinpath(img_name + '_B{}.TIF'.format(i))
        img = cv2.imread(str(img_path), -1)
        rgb.append(img)
    rgb = np.stack(rgb, axis=2)

    # updample rgb
    rgb = rgb.repeat(2, axis=0).repeat(2, axis=1)

    # crop rgb
    rgb = rgb[1: -1, 1: -1]

    # downsample rgb
    rgb = cv2.resize(rgb, None, fx=0.5, fy=0.5)
    rgb = rgb.transpose(2, 0, 1)

    return rgb

def crop(rgb, b8, args, img_name):
    out_path = Path(args.data_dir).joinpath('out')
    out_path.mkdir(exist_ok=True)

    H, W = b8.shape
    Y, X = int(H / args.size), int(W / args.size)

    for y in tqdm(range(Y)):
        for x in range(X):
            rgb_ = rgb[:, y*args.size: (y+1)*args.size, x*args.size: (x+1)*args.size]
            b8_ = b8[y*args.size: (y+1)*args.size, x*args.size: (x+1)*args.size]

            if not (np.zeros((3,1,1), dtype=np.int16) == rgb_).any():
                np.save(out_path.joinpath(img_name + '_rgb_{}_{}.npy'.format(y, x)), rgb_)
                np.save(out_path.joinpath(img_name + '_b8_{}_{}.npy'.format(y, x)), b8_)


def main(args):
    # get base name
    root = Path(args.data_dir)
    img_dir = root.joinpath('img')
    img_names = []
    for img_name in img_dir.iterdir():
        img_names.append(img_name.name.split('_')[0])
    img_names = set(img_names)

    for img_name in img_names:
        print('process {}'.format(img_name))
        # make rgb image
        rgb = make_rgb(args, img_name, img_dir)

        # read b8 band
        b8 = cv2.imread(str(img_dir.joinpath(img_name + '_B8.TIF')), -1)

        # crop images
        crop(rgb, b8, args, img_name)

if __name__ == '__main__':
    start_time = time.time()
    args = get_arguments()
    main(args)
    elapsed_time = time.time() - start_time
    print('elapsed time: {}'.format(elapsed_time))
