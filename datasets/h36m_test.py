import torch
from torch.utils.data import DataLoader

from human36m import HM36Dataset

from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA

import time
import numpy as np
import cv2

def CamProj(x, y, z, fx, fy, u, v, k=1.0):
    cam_x = x / z * fx
    cam_x = cam_x / k + u
    cam_y = y / z * fy
    cam_y = cam_y / k + v
    return cam_x, cam_y

BODY_CONN_COLOR = (
    # pelvis     r_hip           r_knee       r_ank         L_hip            L_knee         L_ank            Spine
    [0, 255, 0], [255, 0, 255], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 128, 255], [204, 204, 255], [255, 102, 255],
    # neck       Head         Site              L_shoulder  L_elbow      L_wrist      R_shoulder   R_elbow
    [255, 0, 0], [0, 255, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
    # R_wrist
    [255, 255, 255]
)

BODY_PARTS_KPT_IDS = [[8, 1],
                      [1, 2],
                      [2, 3],
                      [8, 4],
                      [4, 5],
                      [5, 6],
                      [8, 14],
                      [14, 15],
                      [15, 16],
                      [8, 11],
                      [11, 12],
                      [12, 13],
                      [8, 9],
                      [9, 10],
                      [8, 7],
                      [7, 0],
                      ]
BODY_PARTS_PAF_IDS = ([0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19],
                      [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31])

if __name__ == '__main__':
    st = time.time()

    batch = 1
    dataset = HM36Dataset("d:/datasets/human3.6m", 8, 7, 1, [1], paf_ver=2)
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=1)

    print("load data cost ", time.time() - st)

    blur = GaussianBlur_CUDA(0.9)

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
            img = batch_data['tensor'][i].cuda().unsqueeze(0)
            img = blur(img)
            img = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # img1 = batch_data['tensor1'][i].permute(1, 2, 0).numpy()
            # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

            heatmap = batch_data['heatmap'][i].permute(1, 2, 0).numpy()
            # heatmap = heatmap[:, :, :255].sum(2)
            heatmap = heatmap[:, :, 255:]#.sum(2)
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
                color = BODY_CONN_COLOR[j]
                # color = [0, 255, 255] if j in left else [0, 255, 0]
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

            pafs = batch_data['paf'][i].permute(1, 2, 0).numpy()

            scale = 4
            img_p = np.zeros((pafs.shape[1] * 8, pafs.shape[0] * 8, 3), dtype=np.uint8)
            # pafs[pafs < 0.07] = 0
            for idx in range(len(BODY_PARTS_PAF_IDS)):
                # print(pp, pafs.shape)
                pp = BODY_PARTS_PAF_IDS[idx]
                k_idx = BODY_PARTS_KPT_IDS[idx]
                cc = BODY_CONN_COLOR[idx]

                vx = pafs[:, :, pp[0]]
                vy = pafs[:, :, pp[1]]
                for i in range(pafs.shape[1]):
                    for j in range(pafs.shape[0]):
                        a = (i * 2 * scale, j * 2 * scale)
                        b = (2 * int((i + vx[j, i] * 3) * scale), 2 * int((j + vy[j, i] * 3) * scale))
                        if a[0] == b[0] and a[1] == b[1]:
                            continue

                        cv2.line(img_p, a, b, cc, 1)

                # break

            cv2.imshow("paf", img_p)

            key = cv2.waitKey(0)
            if key == 27:  # esc
                exit(0)
