import torch
from torch.utils.data import DataLoader

from human36m import HM36Dataset

import time
import numpy as np
import cv2

def CamProj(x, y, z, fx, fy, u, v, k=1.0):
    cam_x = x / z * fx
    cam_x = cam_x / k + u
    cam_y = y / z * fy
    cam_y = cam_y / k + v
    return cam_x, cam_y

if __name__ == '__main__':
    st = time.time()

    batch = 2
    dataset = HM36Dataset("d:/datasets/human3.6m", 8, 7, 1)
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=1)

    print("load data cost ", time.time() - st)

    for batch_data in train_loader:
        # print(batch_data)

        for i in range(batch):
            rot = batch_data['R'][i]
            trans = batch_data['T'][i]
            center = batch_data['C'][i]
            focal = batch_data['F'][i]
            scale = batch_data['scale'][i]

            center = center * scale
            focal *= scale

            # print(rot.shape, trans.shape, len(batch_data['joints']))

            img = batch_data['tensor'][i].permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # img1 = batch_data['tensor1'][i].permute(1, 2, 0).numpy()
            # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

            heatmap = batch_data['heatmap'][i].permute(1, 2, 0).numpy()
            heatmap = heatmap[:, :, :255].sum(2)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
            heatmap = cv2.resize(heatmap, (512, 512))

            paf_map = batch_data['paf'][i].permute(1, 2, 0).numpy()
            paf_map = paf_map.sum(2)
            # print(paf_map.shape)
            paf_map = cv2.cvtColor(paf_map, cv2.COLOR_GRAY2BGR)
            paf_map = cv2.resize(paf_map, (512, 512))

            # for xyz in batch_data['joints'][i]:
            #     # print(xyz.shape, xyz, batch_data['joints'][i])
            #     # print(xyz.shape, " - ", trans.shape)
            #     # print(xyz.shape, rot.shape)
            #     xyz = torch.mm(rot,  (xyz - trans).unsqueeze(0).transpose(1, 0)).reshape(3)
            #
            #     x, y = CamProj(xyz[0], xyz[1], xyz[2], focal[0], focal[1], center[0], center[1])
            #     # print(x, y)
            #     cv2.circle(img, (int(x.item()), int(y.item())), 5, [0, 0, 255], -1)

            left = [4, 5, 6, 11, 12, 13]
            for j in range(len(batch_data['joints_2d'][i])):
                xyz = batch_data['joints_2d'][i][j]
                color = [0, 255, 255] if j in left else [0, 255, 0]
                sz = 5 if j in left else 3
                cv2.circle(img, (int(xyz[0].item()), int(xyz[1].item())), sz, color, -1)

            cv2.imshow("test", img)
            cv2.imshow("heatmap", heatmap)
            cv2.imshow("paf", paf_map)

            # print(type(batch_data['extra'][i]), batch_data['extra'][i].shape)
            for j in range(2):
            # for j in range(len(batch_data['extra'][i])):
                t = batch_data['extra'][i][j * 3:(j+1)*3, ...]
                e = t.permute(1, 2, 0).numpy()
                ie = cv2.cvtColor(e, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"extra_{j}", ie)

            key = cv2.waitKey(2000)
            if key == 27:  # esc
                exit(0)
