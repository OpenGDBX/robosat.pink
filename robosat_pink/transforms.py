"""PyTorch-compatible transformations."""

import random
import torch
import cv2
import numpy as np
import random

class ImageToTensor:
    """Callable to convert a NumPy H,W,C image into a PyTorch C,W,H tensor."""

    def __call__(self, image):
        return torch.from_numpy(np.moveaxis(image, 2, 0)).float()


class MaskToTensor:
    """Callable to convert a NumPy H,W image into a PyTorch tensor."""

    def __call__(self, mask):
        return torch.from_numpy(mask).long()


class JointCompose:
    """Callable to transform an image and it's mask at the same time."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask


class JointTransform:
    """Callable to compose non-joint transformations into joint-transformations on images and mask."""

    def __init__(self, image_transform, mask_transform):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


class JointRandomFlipOrRotate:
    """Callable to randomly rotate image and its mask."""

    def __init__(self, p):
        assert p >= 0.0 and p <= 1.0, "Probability must be expressed in 0-1 interval"
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        transform = random.choice(["Rotate90", "Rotate180", "Rotate270", "HorizontalFlip", "VerticalFlip"])

        if transform == "Rotate90":
            return cv2.flip(cv2.transpose(image), +1), cv2.flip(cv2.transpose(mask), +1)
        elif transform == "Rotate180":
            return cv2.flip(image, -1), cv2.flip(mask, -1)
        elif transform == "Rotate270":
            return cv2.flip(cv2.transpose(image), 0), cv2.flip(cv2.transpose(mask), 0)
        elif transform == "HorizontalFlip":
            return cv2.flip(image, +1), cv2.flip(mask, +1)
        elif transform == "VerticalFlip":
            return cv2.flip(image, 0), cv2.flip(mask, 0)


class JointResize:
    """Callable to resize image and its mask."""

    def __init__(self, size):
        self.hw = (size, size)

    def __call__(self, image, mask):

        if self.hw == image.shape[0:2]:
            pass
        elif self.hw[0] < image.shape[0] and self.hw[1] < image.shape[1]:
            image = cv2.resize(image, self.hw, interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, self.hw, interpolation=cv2.INTER_LINEAR)

        if self.hw != mask.shape:
            mask = cv2.resize(mask, self.hw, interpolation=cv2.INTER_NEAREST)

        return image, mask

class JointRandomCrop:
    """Callable to resize image and its mask."""

    def __init__(self, size):
        self.hw = (size, size)

    def __call__(self, image, mask):
        
        if self.hw == image.shape[0:2]:
            pass
        h,w = image.shape[0:2]
        th,tw = self.hw

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        image = image[i:i+th,j:j+tw]
        mask = mask[i:i+th,j:j+tw]

        return image, mask

