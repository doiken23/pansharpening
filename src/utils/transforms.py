import numpy as np

class PansharpenNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        rgb, b8 = data
        data = np.concatenate((rgb, b8), axis=0)

        data = (data - self.mean[:, None, None]) / self.std[:, None, None]

        return (data[:3], data[3][np.newaxis, ...])
