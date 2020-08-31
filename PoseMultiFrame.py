import torch
import torchvision.transforms as T

import numpy as np
import math

from models.with_mobilenet import PoseEstimationWithMobileNetMultiple
# from modules.keypoints import extract_keypoints, group_keypoints
# from modules.pose import Pose, track_poses
from mob_mf_eval import Pose, extract_keypoints, group_keypoints


import collections

def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, min_dims):
    _, _, h, w = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[1] - w - pad[0]))
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[2]))
    # padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
    #                                 cv2.BORDER_CONSTANT, value=pad_value)

    padded_img = torch.nn.functional.pad(img, pad)
    return padded_img, pad


def init_pose(checkpoint_path):
    net = PoseEstimationWithMobileNetMultiple(3, 128, 18, 32)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    net.eval()
    net = net.cuda()

    env = [net]

    return None, env

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio):
    _, _, height, width = img.shape
    scale = net_input_height_size / height


    # scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # scaled_img = normalize(scaled_img, img_mean, img_scale)
    scaled_img = torch.nn.functional.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
    # scaled_img -= 0.5
    # print("SCALED", scaled_img.shape, scaled_img.min(), scaled_img.max())

    min_dims = [net_input_height_size, max(scaled_img.shape[3], net_input_height_size)]
    tensor_img, pad = pad_width(scaled_img, stride, min_dims)


    stages_output = net(tensor_img[:1, ...], tensor_img[1:2, ...], tensor_img[2:3, ...])

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    # heatmaps = torch.nn.functional.interpolate(stage2_heatmaps, scale_factor=upsample_ratio, mode='bicubic', align_corners=False)
    # heatmaps = np.transpose(heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps_tensor = torch.nn.functional.interpolate(stage2_heatmaps, scale_factor=upsample_ratio, mode='bilinear', align_corners=False)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    # pafs = torch.nn.functional.interpolate(stage2_pafs, scale_factor=upsample_ratio, mode='bicubic', align_corners=False)
    # pafs = np.transpose(pafs.squeeze().cpu().data.numpy(), (1, 2, 0))

    return heatmaps, pafs, scale, pad, heatmaps_tensor.squeeze(0)


def anime_frame(rgb, env, size=None, useSigmod=False, useTwice=False):
    if env is None:
        return init_pose("./multiframe_checkpoints/MOBv2.pth")

    net, = env

    stride = 8
    upsample_ratio = 4

    heatmaps, pafs, scale, pad, heatmaps_tensor = infer_fast(net, rgb, 368, stride, upsample_ratio)

    num_keypoints = Pose.num_kpts
    total_keypoints_num = 0
    all_keypoints_by_type = []

    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps_tensor[kpt_idx, :, :], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=19, demo=True)
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
        pose = Pose(pose_keypoints, pose_entries[n][-2])
        current_poses.append(pose)

    # if track:
    #     track_poses(previous_poses, current_poses, smooth=smooth)
    #     previous_poses = current_poses
    # return rgb, env

    # print(rgb.min(), rgb.max())
    # img = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]# * 255
    # img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]# * 255
    # img += 0.5
    # img *= 255
    # img = img.astype(np.uint8)

    img = np.zeros((rgb.shape[2], rgb.shape[3], rgb.shape[1]), dtype=np.uint8)

    show_info = True
    for pose in current_poses:
        pose.draw(img, show_info)
        show_info = False

    # for pose in current_poses:
    #     cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
    #                   (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
    #     if track:
    #         cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
    #                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))


    img = img[:, :, ::-1] #/ 255
    # print(img.shape, img.dtype, img.min(), img.max())
    img = torch.FloatTensor(img.astype(np.float32))
    img /= 255

    img = img.permute(2, 0, 1).unsqueeze(0).cuda()
    output = rgb[:1, ...] * 0.3 + img * 0.7
    # output = img

    return output, env


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

if __name__ == '__main__':
    print("test pose")

    import cv2

    # frame_provider = VideoReader("../AlphaPose/dance3.mp4")
    frame_provider = VideoReader("../AlphaPose/dance2.mp4")

    env = None
    _, env = anime_frame(None, env)
    print('init done')

    frames = []

    for img in frame_provider:

        arr = img[:, :, ::-1].astype(np.float32) / 255
        tensor = torch.FloatTensor(arr).permute(2, 0, 1) #.unsqueeze(0)
        tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor).unsqueeze(0)

        print(tensor.min(), tensor.max())

        # print(tensor.dtype, tensor.shape, tensor.min(), tensor.max())
        if len(frames) == 0:
            for i in range(3):
                t = tensor.clone()
                frames.append(t.cuda())
        else:
            frames.append(tensor.cuda())

        if len(frames) > 3:
            frames.pop(0)

        tensors = torch.cat(frames, 0)

        output, env = anime_frame(tensors, env)

        # print(output.shape)
        img = output.cpu().squeeze(0).permute(1, 2, 0).numpy()[:, :, ::-1]# * 255
        # print(img.shape, img.dtype)
        cv2.imshow('Demo', img)

        key = cv2.waitKey(16)
        if key == 27:  # esc
            break
