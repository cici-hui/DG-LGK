# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import torchvision
import cv2 as cv
from math import sqrt
from skimage import transform
import torchvision.transforms as transforms

import numpy as np

from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')            
        assert img.size == mask.size
        img_nochange = img
        for a in self.augmentations:
            img, mask = a(img, mask)
        
        return np.array(img_nochange), np.array(img), np.array(mask, dtype=np.uint8)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized_and_Crop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.7, 1) * img.size[0])
        h = int(random.uniform(0.7, 1) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(img, mask)


def colorful_spectrum_mix(img1, img2, alpha=0.5, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    # lam = np.random.uniform(0, alpha)
    lam = alpha

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    # img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    # img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    # img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    # img12 = np.uint8(np.clip(img12, 0, 255))

    return img21


class BrightChange(object):
    def __call__(self, img, mask):
        z = random.random()
        if z < 0.1:
            enhancer_bright = ImageEnhance.Brightness(img)
            img_bright = enhancer_bright.enhance(0.75)
            return img_bright, mask
        if 0.1 < z < 0.2:
            enhancer_bright = ImageEnhance.Brightness(img)
            img_bright = enhancer_bright.enhance(0.8)
            return img_bright, mask
        if 0.2 < z < 0.4:
            enhancer_bright = ImageEnhance.Brightness(img)
            img_bright = enhancer_bright.enhance(0.9)
            return img_bright, mask
        if 0.6 < z < 0.8:
            enhancer_bright = ImageEnhance.Brightness(img)
            img_bright = enhancer_bright.enhance(1.1)
            return img_bright, mask
        if 0.8 < z < 0.9:
            enhancer_bright = ImageEnhance.Brightness(img)
            img_bright = enhancer_bright.enhance(1.2)
            return img_bright, mask
        if 0.9 < z < 1:
            enhancer_bright = ImageEnhance.Brightness(img)
            img_bright = enhancer_bright.enhance(1.3)
            return img_bright, mask
        return img, mask


class SharpChange(object):
    def __call__(self, img, mask):
        z = random.random()
        if z < 0.2:
            enhancer_sharp = ImageEnhance.Sharpness(img)
            img_sharp = enhancer_sharp.enhance(0.0)
            return img_sharp, mask
        if 0.6 < z < 0.8:
            enhancer_sharp = ImageEnhance.Sharpness(img)
            img_sharp = enhancer_sharp.enhance(2.0)
            return img_sharp, mask
        if 0.8 < z < 1:
            enhancer_sharp = ImageEnhance.Sharpness(img)
            img_sharp = enhancer_sharp.enhance(3.0)
            return img_sharp, mask
        return img, mask


class ContrastChange(object):
    def __call__(self, img, mask):
        z = random.random()
        if z < 0.3:
            enhancer_contrast = ImageEnhance.Contrast(img)
            img_contrast = enhancer_contrast.enhance(0.8)
            return img_contrast, mask
        if 0.7 < z < 0.9:
            enhancer_contrast = ImageEnhance.Contrast(img)
            img_contrast = enhancer_contrast.enhance(1.2)
            return img_contrast, mask
        if 0.9 < z < 1:
            enhancer_contrast = ImageEnhance.Contrast(img)
            img_contrast = enhancer_contrast.enhance(1.3)
            return img_contrast, mask
        return img, mask


class ColorChange(object):
    def __call__(self, img, mask):
        z = random.random()
        if z < 0.15:
            enhancer_color = ImageEnhance.Color(img)
            img_color = enhancer_color.enhance(0.1)
            return img_color, mask
        if 0.15 < z < 0.25:
            enhancer_color = ImageEnhance.Color(img)
            img_color = enhancer_color.enhance(0.3)
            return img_color, mask
        if 0.25 < z < 0.4:
            enhancer_color = ImageEnhance.Color(img)
            img_color = enhancer_color.enhance(0.5)
            return img_color, mask
        if 0.6 < z < 0.75:
            enhancer_color = ImageEnhance.Color(img)
            img_color = enhancer_color.enhance(1.5)
            return img_color, mask
        if 0.75 < z < 0.85:
            enhancer_color = ImageEnhance.Color(img)
            img_color = enhancer_color.enhance(1.7)
            return img_color, mask
        if 0.85 < z < 1:
            enhancer_color = ImageEnhance.Color(img)
            img_color = enhancer_color.enhance(2.0)
            return img_color, mask
        return img, mask