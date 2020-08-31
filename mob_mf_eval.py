import os
import torch
from models.with_mobilenet import PoseEstimationWithMobileNetMultiple
import torchvision.transforms as T

from PIL import Image

import math
import numpy as np
from operator import itemgetter

import cv2

from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA

from modules.one_euro_filter import OneEuroFilter

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

BODY_CONN_COLOR = (
    [0, 255, 0], [255, 0, 255], [0, 0, 255], [0, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0],
    [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]
)


class Pose:
    num_kpts = 17
    # kpt_names = ['nose', 'neck',
    #              'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
    #              'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
    #              'r_eye', 'l_eye',
    #              'r_ear', 'l_ear']

    hr_kpt_names = ['Pelvis', 'R_hip',
                    'R_knee', 'R_ank', 'L_hip', 'L_knee', 'L_ank', 'Spine',
                    'Neck', 'Head', 'Site', 'L_shoulder', 'L_elbow', 'L_wrist',
                    'R_shoulder', 'R_elbow',
                    'R_wrist']

    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img, show_names=False):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        kpts = dict()
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)

                if show_names:
                    kpts[Pose.hr_kpt_names[kpt_a_id]] = [x_a, y_a]
                    kpts[Pose.hr_kpt_names[kpt_b_id]] = [x_b, y_b]

        px = img.shape[1] - 200
        py = 20

        for k, v in kpts.items():
            # draw part coordination as text
            t = f"{k}: {v[0]}, {v[1]}"
            cv2.putText(img, t,
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255)
                        )
            # print("put text ", t, " at ", px, py)
            py += 20


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


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
        keypoint_with_score_and_id = (keypoints[i][0].float().item(), keypoints[i][1].float().item(), heatmap[keypoints[i][1], keypoints[i][0]].float().item(),
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num

def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False):
    pose_entries = []
    # print("group keypoints", pafs.shape)

    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]#.cpu().numpy()
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]#.cpu().numpy()
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1                   # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]        # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array([kpts_a[i][0], kpts_a[i][1]], dtype=np.float32) #.cpu().numpy())
            for j in range(num_kpts_b):
                # kpt_b = np.array(kpts_b[j][0:2])# .cpu().numpy())
                kpt_b = np.array([kpts_b[j][0], kpts_b[j][1]], dtype=np.float32)# .cpu().numpy())
                mid_point = [(), ()]
                mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                   vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])

                height_n = pafs.shape[0] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        if not demo:
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                        else:
                            px = int(x[point_idx])
                            py = int(y[point_idx])
                        paf = part_pafs[py, px, 0:2]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        # elif part_id == 14 or part_id == 15:
        #     kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        #     kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        #     for i in range(len(connections)):
        #         for j in range(len(pose_entries)):
        #             if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
        #                 pose_entries[j][kpt_b_id] = connections[i][1]
        #             elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
        #                 pose_entries[j][kpt_a_id] = connections[i][0]
        #     continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


if __name__ == '__main__':
    print("eval")
    torch.set_grad_enabled(False)

    batch = 16
    num_refinement_stages = 3

    net = PoseEstimationWithMobileNetMultiple(3, 128, 18, 32)
    checkpoint_fn = "./multiframe_checkpoints/MOBv3.pth"
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

    # t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)
    t -= 0.5

    t = t.cuda().unsqueeze(0)
    t = torch.nn.functional.interpolate(t, (368, 368), mode='bilinear', align_corners=False)
    # print(t.shape)

    blur = GaussianBlur_CUDA(0.5)
    # t = blur(t)

    t1 = t.clone()
    t2 = t.clone()

    outputs = net(t, t1, t2)

    heatmaps = outputs[-2]#.squeeze(0)
    paf_maps = outputs[-1]#.squeeze(0)

    print(heatmaps.shape, paf_maps.shape)

    #todo: resize to training size
    heatmaps = torch.nn.functional.interpolate(heatmaps, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)
    paf_maps = torch.nn.functional.interpolate(paf_maps, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(17):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[kpt_idx, ...], all_keypoints_by_type, total_keypoints_num)

    pafs = paf_maps.squeeze(0).cpu().permute(1, 2, 0).numpy()
    stride = 8
    upsample_ratio = 4
    scale = 1
    num_keypoints = 17

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=19, demo=True)
    print("POSE ENTRIES count", len(pose_entries), " total keypoints ", total_keypoints_num)
    print(*pose_entries, sep="\n")
    print(*all_keypoints_by_type, sep="\n")

    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[
            1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[
            0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        print("pose entry", len(pose_entries[n]), pose_entries[n])
        pose = Pose(pose_keypoints, pose_entries[n][-2])
        current_poses.append(pose)

    # print(all_keypoints_by_type)
    print("Pose count", len(current_poses))

    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # for j in range(total_keypoints_num):
    for pp in all_keypoints_by_type:
        for xyz in pp:
            print("XYZ", xyz[0], xyz[1])

            color = [0, 255, 255] if xyz[2] < 0.5 else [0, 255, 0]
            sz = 3 if xyz[2] < 0.5 else 5
            cv2.circle(img, (int(xyz[0] * 2), int(xyz[1] * 2)), sz, color, -1)

    print("predict count", total_keypoints_num)

    cv2.imshow("test", img)

    img_s = np.zeros((368, 368, 3), dtype=np.uint8)

    show_info = True
    for pose in current_poses:
        pose.draw(img_s, show_info)
        show_info = False

    img = img * 0.3 + img_s * 0.7
    cv2.imshow("skel", img)
    cv2.imshow("paf_sum", pafs.sum(2))

    scale = 4
    img_p = np.zeros((368 * scale, 368 * scale, 3), dtype=np.uint8)
    pafs[pafs < 0.07] = 0
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
