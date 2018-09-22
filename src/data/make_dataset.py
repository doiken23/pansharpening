import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import tifffile
import joblib
from tqdm import tqdm
from pprint import pprint

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('size', type=int)
    parser.add_argument('--test', action='store_true', default=False)
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
    rgb = rgb.repeat(4, axis=0).repeat(4, axis=1)

    # crop rgb
    rgb = rgb[1: -1, 1: -1]

    # downsample rgb
    rgb = cv2.resize(rgb, None, fx=0.5, fy=0.5)
    rgb = rgb.transpose(2, 0, 1)

    return rgb

def crop(rgb, b8, args, img_name):
    if args.test:
        out_path = Path(args.data_dir).joinpath('out', 'test')
    else:
        out_path = Path(args.data_dir).joinpath('out', 'train')
    out_path.mkdir(exist_ok=True)

    if b8.shape != rgb.shape[1:]:
        print('Size of RGB abd B8 is different!!!')
    H, W = b8.shape
    Y, X = int(H / args.size), int(W / args.size)

    # initialize statistics
    maxs = []
    mins = []
    means = []
    stds = []

    n = 0
    for y in tqdm(range(Y)):
        for x in range(X):
            rgb_ = rgb[:, y*args.size: (y+1)*args.size, x*args.size: (x+1)*args.size]
            b8_ = b8[y*args.size: (y+1)*args.size, x*args.size: (x+1)*args.size]

            if not (np.zeros((3,1,1), dtype=np.int16) == rgb_).any() \
                    and not (np.concatenate((rgb_, b8_[np.newaxis, ...])) > 15000).any():
                n += 1
                data = {'rgb': rgb_,
                        'b8': b8_}
                joblib.dump(data, out_path.joinpath(img_name + '_{}_{}.pkl'.format(y, x)))

                img = np.concatenate((rgb_, b8_[np.newaxis, ...]), axis=0)
                maxs.append(np.max(img, axis=(1,2)))
                mins.append(np.min(img, axis=(1,2)))
                means.append(np.mean(img, axis=(1,2)))
                stds.append(np.std(img, axis=(1,2)))
    print('{} / {} images are saved.'.format(n, (x+1)*(y+1)))
    
    max = np.max(maxs, axis=0)
    min = np.min(mins, axis=0)
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return max, min, mean, std

def main(args):
    # get base name
    root = Path(args.data_dir)
    if args.test:
        img_dir = root.joinpath('img', 'test')
    else:
        img_dir = root.joinpath('img', 'train')
    img_names = []
    for img_name in img_dir.iterdir():
        img_names.append(img_name.name.split('_')[0])
    img_names = set(img_names)

    # initialize statistics
    maxs = []
    mins = []
    means = []
    stds = []

    for img_name in img_names:
        print('===== process {} ====='.format(img_name))
        # make rgb image
        rgb = make_rgb(args, img_name, img_dir)

        # read b8 band
        b8 = cv2.imread(str(img_dir.joinpath(img_name + '_B8.TIF')), -1)

        # crop images
        max, min, mean, std = crop(rgb, b8, args, img_name)
        maxs.append(max)
        mins.append(min)
        means.append(mean)
        stds.append(std)

    # save statstics
    max = np.max(maxs, axis=0)
    min = np.min(mins, axis=0)
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    if not args.test:
        Path(args.data_dir).joinpath('statistic').mkdir(exist_ok=True)
        statistic_path = Path(args.data_dir).joinpath('statistic', 'statistic.pkl')
        statistic = {'max': max,
                'min': min,
                'mean': mean,
                'std': std}
        print('===== statistics =====')
        pprint(statistic)
        joblib.dump(statistic, statistic_path)

if __name__ == '__main__':
    start_time = time.time()
    args = get_arguments()
    main(args)
    elapsed_time = time.time() - start_time
    print('elapsed time: {}'.format(elapsed_time))
