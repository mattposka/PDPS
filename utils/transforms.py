import random

import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image, ImageFilter


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


def im2vl(img):
    img_tmp = np.ones(img.shape[:2], dtype=np.uint8)
    neg_msk = np.all(img == [255, 255, 255], axis=2)
    img_tmp[neg_msk] = 0
    return img_tmp


def vl2im(img):
    img_tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    neg_msk = img == 0
    pos_msk = img == 1
    img_tmp[neg_msk] = [255, 255, 255]
    img_tmp[pos_msk] = [255, 0, 0]
    return img_tmp
