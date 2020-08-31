import os
import torch
import torchvision.transforms as T

from PIL import Image

import math
import numpy as np
from operator import itemgetter

import cv2

def get_motion_blur_kernel(kernel_size, angle):
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2) , angle, 1)
    motion_kernel = np.diag(np.ones(kernel_size))
    motion_kernel = cv2.warpAffine(motion_kernel, M, (kernel_size, kernel_size))

    return torch.from_numpy(motion_kernel).float() / kernel_size

def rand_motion_blur_weight(channels, max_kernel_size, max_angle):
    ks = 3 + np.random.randint(0, max_kernel_size) * 2
    angle = np.random.sample() * max_angle
    kernel = get_motion_blur_kernel(ks, angle)

    weight = torch.zeros(channels, channels, ks, ks)
    for i in range(channels):
        weight[i, i] = kernel

    return weight


if __name__ == '__main__':
    print("test motion blur")

    input_fn = "D:/datasets/coco/val2017/000000000785.jpg"
    t = T.ToTensor()(Image.open(input_fn)).unsqueeze(0)

    ww = rand_motion_blur_weight(3, 9, 360)
    t = torch.nn.functional.conv2d(t, ww)

    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("test", img)
    key = cv2.waitKey(0)
    if key == 27:  # esc
        exit(0)