import torch
from torch.utils.data import DataLoader

from datasets.human36m import HM36Dataset

import time
import numpy as np
import cv2


if __name__ == '__main__':
    st = time.time()

    batch = 2
    dataset = HM36Dataset("d:/datasets/human3.6m", 8, 7, 1)
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=1)

    print("load data cost ", time.time() - st)

    for batch_data in train_loader:
        # print(type(batch_data['tensor']), type(batch_data['extra']))
        print(batch_data['tensor'].shape, batch_data['extra'].shape)
