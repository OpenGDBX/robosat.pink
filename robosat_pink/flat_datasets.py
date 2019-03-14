"""PyTorch-compatible datasets. Cf: https://pytorch.org/docs/stable/data.html """

import os
import glob
import sys

import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data

class DatasetFiles(torch.utils.data.Dataset):
    """Dataset for images stored in slippy map format."""

    def __init__(self, root, mode, transform=None):
        super().__init__()

        self.files = []
        self.transform = transform

        root = os.path.expanduser(root)
        self.files = [path for path in glob.glob(os.path.join(root, "*.*"))]
        self.files.sort(key=lambda file: file)
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]

        if self.mode == "mask":
            image = np.array(Image.open(path).convert("P"))

        elif self.mode == "image":
            path = os.path.expanduser(path)
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)

        if self.transform is not None:
            image = self.transform(image)

        return image


class DatasetFilesConcat(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format."""

    def __init__(self, path, channels, target, joint_transform=None):
        super().__init__()

        assert len(channels)
        self.channels = channels
        self.inputs = dict()

        for channel in channels:
            for band in channel["bands"]:
                self.inputs[channel["sub"]] = DatasetFiles(os.path.join(path, channel["sub"]), mode="image")

        self.target = DatasetFiles(target, mode="mask")
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):

        mask = self.target[i]

        for channel in self.channels:
            try:
                data, band_tile = self.inputs[channel["sub"]][i]
                assert band_tile == tile

                for band in channel["bands"]:
                    data_band = data[:, :, int(band) - 1] if len(data.shape) == 3 else []
                    data_band = data_band.reshape(mask.shape[0], mask.shape[1], 1)
                    tensor = np.concatenate((tensor, data_band), axis=2) if "tensor" in locals() else data_band  # noqa F821
            except:
                sys.exit("Unable to concatenate input Tensor")

        if self.joint_transform is not None:
            tensor, mask = self.joint_transform(tensor, mask)

        return tensor, mask
