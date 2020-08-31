import os
import torch
from models.with_resnet34 import PoseEstimationWithResnet50Single
import torchvision.transforms as T

from PIL import Image

import math
import numpy as np
from operator import itemgetter

import cv2

from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA

def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    # print(heatmap.shape)
    # heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_with_borders = torch.nn.functional.pad(heatmap, [2, 2, 2, 2], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)

    # print("peaks ", heatmap_peaks)

    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    # keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)

    keypoints = list(zip(torch.nonzero(heatmap_peaks, as_tuple=True)[1], torch.nonzero(heatmap_peaks, as_tuple=True)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    # suppressed = np.zeros(len(keypoints), np.uint8)
    suppressed = torch.zeros(len(keypoints))
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


if __name__ == '__main__':
    print("eval")
    torch.set_grad_enabled(False)

    batch = 16
    num_refinement_stages = 3

    net = PoseEstimationWithResnet50Single()
    checkpoint_fn = "./resnet50_single_checkpoints/checkpoint_iter_500.pth"
    # checkpoint_fn = "./multiframe_checkpoints/checkpoint_iter_10000.pth"
    net.load_state_dict(torch.load(checkpoint_fn)['state_dict'])
    net = net.cuda()
    net.eval()

    print("net loaded")
    input_fn = "D:/datasets/coco/val2017/000000000785.jpg"
    # input_fn = "D:/datasets/mh.jpg"
    # input_fn = "D:/datasets/coco/val2017/000000000776.jpg"
    # input_fn = "D:/datasets/human3.6m/images/s_11_act_16_subact_02_ca_03/s_11_act_16_subact_02_ca_03_000003.jpg"

    t = T.ToTensor()(Image.open(input_fn))

    _, oh, ow = t.shape
    sz = max(oh, ow)
    print("input size", ow, oh, sz)

    pad = []
    pad.append((sz - ow) // 2)
    pad.append(sz - ow - pad[0])
    pad.append((sz - oh) // 2)
    pad.append(sz - oh - pad[2])

    t = torch.nn.functional.pad(t, pad, mode='constant', value=0.5)

    t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)

    t = t.cuda().unsqueeze(0)
    t = torch.nn.functional.interpolate(t, (256, 256), mode='bilinear', align_corners=False)
    print(t.shape)

    blur = GaussianBlur_CUDA(0.5)
    # t = blur(t)

    t1 = t.clone()
    t2 = t.clone()

    outputs = net(t)

    heatmaps = outputs#.squeeze(0)
    # paf_maps = outputs[-1]#.squeeze(0)
    print("heatmap shape", heatmaps.shape)

    #todo: resize to training size
    heatmaps = torch.nn.functional.interpolate(heatmaps, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)
    # paf_maps = torch.nn.functional.interpolate(paf_maps, scale_factor=8, mode='bicubic', align_corners=False).squeeze(0)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(17):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[kpt_idx, ...], all_keypoints_by_type, total_keypoints_num)

    # print(all_keypoints_by_type)

    heatmap = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
    heatmap = heatmap[:, :, :17].sum(2)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.resize(heatmap, (512, 512))

    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # for j in range(total_keypoints_num):
    for pp in all_keypoints_by_type:
        for xyz in pp:
            print("XYZ", xyz[0].item(), xyz[1].item())

            color = [0, 255, 255] if xyz[2].item() < 0.5 else [0, 255, 0]
            sz = 3 if xyz[2].item() < 0.5 else 5
            cv2.circle(img, (int(xyz[0].item()), int(xyz[1].item())), sz, color, -1)

    print("predict count", total_keypoints_num)

    cv2.imshow("test", img)
    cv2.imshow("heatmap", heatmap)
    key = cv2.waitKey(0)
    if key == 27:  # esc
        exit(0)
