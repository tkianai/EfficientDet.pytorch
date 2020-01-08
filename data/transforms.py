import torch
import torchvision
from torchvision import transforms as ttrans
import cv2
import numpy as np


class Normalize(object):

    def __init__(self):
        self.mean = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.std = np.array([[[1.0, 1.0, 1.0]]])

    def __call__(self, sample):
        image = sample['image']
        image = (image.astype(np.float32) - self.mean) / self.std

        sample['image'] = image

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            image, bboxes = sample['image'], sample['bboxes']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = bboxes[:, 0].copy()
            x2 = bboxes[:, 2].copy()

            bboxes[:, 0] = cols - x2
            bboxes[:, 2] = cols - x1

            sample['image'] = image
            sample['bboxes'] = bboxes

        return sample


class RandomVerticalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            image, bboxes = sample['image'], sample['bboxes']
            image = image[::-1, :, :]

            rows, cols, channels = image.shape

            y1 = bboxes[:, 1].copy()
            y2 = bboxes[:, 3].copy()

            bboxes[:, 1] = rows - y2
            bboxes[:, 3] = rows - y1

            sample['image'] = image
            sample['bboxes'] = bboxes

        return sample


class Resize(object):

    def __init__(self, max_size=512):
        self.max_size = max_size

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']
        height, width, _ = image.shape
        if height > width:
            scale = self.max_size / height
            resized_height = self.max_size
            resized_width = int(width * scale)
        else:
            scale = self.max_size / width
            resized_height = int(height * scale)
            resized_width = self.max_size

        image = cv2.resize(image, (resized_width, resized_height))
        new_image = np.zeros((self.max_size, self.max_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        bboxes *= scale
        sample['image'] = torch.from_numpy(new_image)
        sample['bboxes'] = torch.from_numpy(bboxes)
        sample['labels'] = torch.from_numpy(sample['labels'])
        sample['scale'] = scale

        return sample



def build_transforms(is_train=True, inp_size=512):
    if is_train:
        transform = ttrans.Compose([
            Normalize(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(prob=0.0),
            Resize(inp_size),
        ])
    else:
        transform = ttrans.Compose([
            Normalize(),
            Resize(inp_size),
        ])

    return transform
