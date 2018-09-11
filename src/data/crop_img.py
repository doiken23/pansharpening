import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import joblib
from tqdm import tqdm

def crop_img(data_dir, base_name, out, size=224):
    img_path = Path(data_dir).out('rgb').joinpath(base_name + '_RGB.tif')
    img = tifffile.imread(str(img_path))
    mask_path = Path(data_dir).out('mask').joinpath(base_name + '_MASK.png')
    mask = np.array(Image.open(mask_path))
    b8_path = Path(data_dir).out('B8').joinpath(base_name + '_B8.TIF')
    b8 = tifffile.imread(str(b8_path))
    b8 = b8[np.newaxis, ...]

    masked_img = (img * mask)
    img = masked_img

    h, w = img.shape[1:]
    H, W = int(h / size), int(w / size)
    for y in tqdm(range(H)):
        for x in range(W):
            img_ = img[:, y*size: (y+1)*size, x*size: (x+1)*size]
            if not (np.zeros((3, 1, 1), dtype=np.uint16) == img_).any():
                b8_ = b8[:, 2*y*size: 2*(y+1)*size, 2*x * size: 2*(x+1)*size]
                data = {
                        'rgb': img_,
                        'b8': b8_
                        }
                out_path = Path(out).out(base_name + '_{}_{}.pkl'.format(y, x))
                joblib.dump(data, out_path)

def main():
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    # parse data_dir
    data_dir = Path(args.data_dir)
    rgb_dir = data_dir.out('rgb')
    base_names = [rgb_img.stem.split('_')[0] for rgb_img in rgb_dir.iterdir()]

    # crop images
    for base_name in tqdm(base_names):
        crop_img(data_dir, base_name, args.out, size=args.size)

if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print('elapsed time: {} [s]'.format(elapsed_time))
