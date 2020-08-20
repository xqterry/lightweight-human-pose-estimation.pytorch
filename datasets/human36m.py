import torch
import json
from torch.utils.data.dataset import Dataset

import numpy as np
import os

from PIL import Image
import torchvision.transforms as T

import cv2
import math

def CamProj(x, y, z, fx, fy, u, v, k=1.0):
    cam_x = x / z * fx
    cam_x = cam_x / k + u
    cam_y = y / z * fy
    cam_y = cam_y / k + v
    return cam_x, cam_y

def CamBackProj(cam_x, cam_y, depth, fx, fy, u, v):
    x = (cam_x - u) / fx * depth
    y = (cam_y - v) / fy * depth
    z = depth
    return x, y, z


def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)


def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def cvt_ThetaToM(theta, w, h, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.

    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required

    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv[:2, :]
    return M[:2, :]

import random

class HM2DRotate:
    def __init__(self, pad, max_rotate_degree=40):
        self._pad = pad
        self._max_rotate_degree = max_rotate_degree

    def __call__(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self._max_rotate_degree
        _, h, w = sample['tensor'].shape

        img_center = sample['C'] * sample['scale']
        R = cv2.getRotationMatrix2D((img_center[0], img_center[1]), degree, 1)

        # print(type(sample['C']), type(img_center), img_center.shape, theta.shape)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h)



        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]

        theta = cvt_MToTheta(R, w, h)
        theta = torch.FloatTensor(theta)

        grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), torch.Size([1, 3, bound_h, bound_w]), align_corners=False)
        tensor1 = torch.nn.functional.grid_sample(sample['tensor'].unsqueeze(0), grid, align_corners=False).squeeze(0)


        img = sample['tensor'].permute(1, 2, 0).numpy()[:, :, [2, 1, 0]]
        img = cv2.warpAffine(img, R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)

        tensor = torch.Tensor(img).permute(2, 0, 1)[[2, 1, 0], :, :]

        for i in range(len(sample['extra'])):
            t = sample['extra'][i]
            img = t.permute(1, 2, 0).numpy()[:, :, [2, 1, 0]]
            img = cv2.warpAffine(img, R, dsize=dsize,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)

            sample['extra'][i] = torch.Tensor(img).permute(2, 0, 1)[[2, 1, 0], :, :]

        # print(tensor.shape, tensor1.shape)

        # sample['label']['img_height'], sample['label']['img_width'], _ = sample['image'].shape
        # sample['mask'] = cv2.warpAffine(sample['mask'], R, dsize=dsize,
        #                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))  # border is ok
        # label = sample['label']
        # label['objpos'] = self._rotate(label['objpos'], R)
        keypoints = sample['joints_2d']
        for keypoint in keypoints:
            point = [keypoint[0], keypoint[1]]
            point = self._rotate(point, R)
            keypoint[0], keypoint[1] = point[0], point[1]

        sample['joints_2d'] = keypoints
        sample['tensor'] = tensor
        # sample['tensor1'] = tensor1

        return sample

    def _rotate(self, point, R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]


class HM2DCropPad:
    def __init__(self, pad=[0.5, 0.5, 0.5], center_perterb_max=50, crop_x=448, crop_y=448):
        self._pad = pad
        self._center_perterb_max = center_perterb_max
        self._crop_x = crop_x
        self._crop_y = crop_y

    def __call__(self, sample):
        prob_x = random.random()
        prob_y = random.random()

        offset_x = int((prob_x - 0.5) * 2 * self._center_perterb_max)
        offset_y = int((prob_y - 0.5) * 2 * self._center_perterb_max)

        obj_center = sample['joints_2d'][0]
        shifted_center = (obj_center[0] + offset_x, obj_center[1] + offset_y)
        offset_left = -int(shifted_center[0] - self._crop_x / 2)
        offset_up = -int(shifted_center[1] - self._crop_y / 2)

        c, h, w = sample['tensor'].shape
        cropped_image = torch.ones(c, self._crop_y, self._crop_x) * 0.5

        image_x_start = int(shifted_center[0] - self._crop_x / 2)
        image_y_start = int(shifted_center[1] - self._crop_y / 2)
        image_x_finish = image_x_start + self._crop_x
        image_y_finish = image_y_start + self._crop_y
        crop_x_start = 0
        crop_y_start = 0
        crop_x_finish = self._crop_x
        crop_y_finish = self._crop_y

        should_crop = True
        if image_x_start < 0:  # Adjust crop area
            crop_x_start -= image_x_start
            image_x_start = 0
        if image_x_start >= w:
            # should_crop = False
            image_x_start = w - self._crop_x
            image_x_finish = w
            offset_left = - (obj_center[0] - image_x_start + self._crop_x / 2)

        if image_y_start < 0:
            crop_y_start -= image_y_start
            image_y_start = 0
        if image_y_start >= h:
            # should_crop = False
            image_y_start = h - self._crop_y
            image_y_finish = h
            offset_up = - (obj_center[1] - image_y_start + self._crop_x / 2)

        if image_x_finish > w:
            diff = image_x_finish - w
            image_x_finish -= diff
            crop_x_finish -= diff
        if image_x_finish < 0:
            # should_crop = False
            image_x_start = 0
            image_x_finish = self._crop_x

        if image_y_finish > h:
            diff = image_y_finish - h
            image_y_finish -= diff
            crop_y_finish -= diff
        if image_y_finish < 0:
            image_y_start = 0
            image_y_finish = self._crop_y
            # should_crop = False

        if should_crop:
            cropped_image[:, crop_y_start:crop_y_finish, crop_x_start:crop_x_finish] = \
                sample['tensor'][:, image_y_start:image_y_finish, image_x_start:image_x_finish]

            for i in range(len(sample['extra'])):
                t = torch.ones(c, self._crop_y, self._crop_x) * 0.5
                t[:, crop_y_start:crop_y_finish, crop_x_start:crop_x_finish] = \
                    sample['extra'][i][:, image_y_start:image_y_finish, image_x_start:image_x_finish]
                sample['extra'][i] = t

        keypoints = sample['joints_2d']

        for keypoint in keypoints:
            keypoint[0] += offset_left
            keypoint[1] += offset_up

        sample['tensor'] = cropped_image
        sample['joints_2d'] = keypoints

        return sample

    def _inside(self, point, width, height):
        if point[0] < 0 or point[1] < 0:
            return False
        if point[0] >= width or point[1] >= height:
            return False
        return True


class HM2DScale:
    def __init__(self, prob=1, min_scale=0.5, max_scale=1.2, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_dist = target_dist

    def __call__(self, sample):
        keypoints = sample['joints_2d']
        main_tensor = sample['tensor']

        tensor = torch.cat([main_tensor, *sample['extra']], 0)

        prob = np.random.sample()
        scale = (self._max_scale - self._min_scale) * prob + self._min_scale

        for keypoint in keypoints:
            keypoint[0] *= scale
            keypoint[1] *= scale
            keypoint[2] *= scale

        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)

        sample['tensor'] = tensor[:3, :, :]
        sample['extra'] = [tensor[3:6, :, :], tensor[6:, :, :]]
        sample['joints_2d'] = keypoints
        sample['scale'] = scale

        return sample

class HM2DFlip:
    def __init__(self, prob=0.5):
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        do_flip = prob <= self._prob
        if not do_flip:
            return sample

        # sample['image'] = cv2.flip(sample['image'], 1)
        # sample['mask'] = cv2.flip(sample['mask'], 1)
        sample['tensor'] = torch.flip(sample['tensor'], [2])

        for i in range(len(sample['extra'])):
            t = sample['extra'][i]
            sample['extra'][i] = torch.flip(t, [2])

        _, h, w = sample['tensor'].shape
        keypoints = sample['joints_2d']
        for keypoint in keypoints:
            keypoint[0] = w - 1 - keypoint[0]
        sample['joints_2d'] = self._swap_left_right(keypoints)

        return sample

    def _swap_left_right(self, keypoints):
        right = [1, 2, 3, 14, 15, 16]
        left = [4, 5, 6, 11, 12, 13]
        # return keypoints
        for r, l in zip(right, left):
            tl, tr = keypoints[l].copy(), keypoints[r]
            keypoints[l] = tr
            keypoints[r] = tl
            # keypoints[r], keypoints[l] = keypoints[l], keypoints[r]
        return keypoints

def create_heatmap(keypoints, h, w, stride, sigma):
    n_keypoints = 256
    tensor = torch.zeros(n_keypoints, h // stride, w // stride)

    n_sigma = 4

    for i in range(len(keypoints)):
        p = keypoints[i]
        x, y, z, v = p
        if v <= 1:
            tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
            tl[0] = max(tl[0], 0)
            tl[1] = max(tl[1], 0)

            br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
            _, map_h, map_w = tensor.shape
            br[0] = min(br[0], map_w * stride)
            br[1] = min(br[1], map_h * stride)

            shift = stride / 2 - 0.5
            for map_y in range(tl[1] // stride, br[1] // stride):
                for map_x in range(tl[0] // stride, br[0] // stride):
                    d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                         (map_y * stride + shift - y) * (map_y * stride + shift - y)
                    exponent = d2 / 2 / sigma / sigma
                    if exponent > 4.6052:  # threshold, ln(100), ~0.01
                        continue

                    tensor[i, map_y, map_x] += math.exp(-exponent)
                    if tensor[i, map_y, map_x] > 1:
                        tensor[i, map_y, map_x] = 1

    tensor[-1] = 1 - tensor.max(0).values

    return tensor

def set_paf(tensor, a, b, stride, thickness):
    x_a, y_a, _, _ = a
    x_b, y_b, _, _ = b

    x_a /= stride
    y_a /= stride
    x_b /= stride
    y_b /= stride
    x_ba = x_b - x_a
    y_ba = y_b - y_a
    _, h_map, w_map = tensor.shape
    x_min = int(max(min(x_a, x_b) - thickness, 0))
    x_max = int(min(max(x_a, x_b) + thickness, w_map))
    y_min = int(max(min(y_a, y_b) - thickness, 0))
    y_max = int(min(max(y_a, y_b) + thickness, h_map))
    norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
    if norm_ba < 1e-7:  # Same points, no paf
        return
    x_ba /= norm_ba
    y_ba /= norm_ba

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            x_ca = x - x_a
            y_ca = y - y_a
            d = math.fabs(x_ca * y_ba - y_ca * x_ba)
            if d <= thickness:
                tensor[0, y, x] = x_ba
                tensor[1, y, x] = y_ba

    return tensor


def create_paf_map(keypoints, h, w, stride, thickness):
    PAF_GROUP_IDX = [[8, 9],
                     [9, 10],
                     [8, 7],
                     [7, 0],
                     [0, 1],
                     [1, 2],
                     [2, 3],
                     [0, 4],
                     [4, 5],
                     [5, 6],
                     [8, 14],
                     [14, 15],
                     [15, 16],
                     [8, 11],
                     [11, 12],
                     [12, 13],
                     ]
    paf_map = torch.zeros(len(PAF_GROUP_IDX) * 2, h // stride, w // stride)
    for i in range(len(PAF_GROUP_IDX)):
        pp = PAF_GROUP_IDX[i]
        a = keypoints[pp[0]]
        b = keypoints[pp[1]]
        if a[3] <= 1 and b[3] <= 1:
            set_paf(paf_map[i:i+2, :, :], a, b, stride, thickness)

    return paf_map


class HM36Dataset(Dataset):
    def __init__(self, data_folder, stride, sigma, paf_thickness, transform=None):
        super().__init__()
        self._data_folder = data_folder
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness
        self._transform = transform

        # self.train_list = [1, 5, 6, 7, 8, 9]
        self.train_list = [1]

        self.cameras = self._load_cameras()

        all_counts = []
        all_data = []
        all_joints = []
        for d_id in self.train_list:
            data_fn = os.path.join(self._data_folder, "annotations/annotations", f"Human36M_subject{d_id}_data.json")
            joint_fn = os.path.join(self._data_folder, "annotations/annotations", f"Human36M_subject{d_id}_joint_3d.json")
            with open(data_fn, 'rb') as f:
                data = json.load(f)
                all_data.append(data)
                all_counts.append(len(data['images']))
                f.close()

            with open(joint_fn, "rb") as f:
                jj = json.load(f)
                all_joints.append(jj)
                f.close()

        self.all_counts = all_counts
        self.all_data = all_data
        self.all_joints = all_joints

        self._transform = T.Compose([
            HM2DScale(),
            HM2DRotate(pad=(0.5, 0.5, 0.5)),
            HM2DCropPad(),
            HM2DFlip(0.5)
            # CropPad(pad=(128, 128, 128)),
            # Flip()
        ])


    def _load_camera_params(self, sub_id, cam_id):
        k_sub = f"subject{sub_id}"
        k_cam = f"camera{cam_id}"

        rot = np.array(self.cameras[k_sub][k_cam]['R']).transpose()
        trans = np.array(self.cameras[k_sub][k_cam]['T']).reshape((3))
        c = np.array(self.cameras[k_sub][k_cam]['c']).reshape((2))
        f = np.array(self.cameras[k_sub][k_cam]['f']).reshape((2))

        return rot, trans, c, f

    def _load_cameras(self):
        cam_fn = os.path.join(self._data_folder, "annotations/h36m/cameras.json")
        with open(cam_fn, "rb") as fp:
            cams = json.load(fp)
            fp.close()

        return cams

    def _load_extra_frames(self, curr_data, data_idx, idx, count=2):
        frames = []
        step = 1 + int(5 * np.random.sample())
        total = len(self.all_data[data_idx]['images'])
        last_data = curr_data
        for i in range(count):
            idx = idx + step
            if idx >= total:
                frames.append(last_data)
            else:
                img_info = self.all_data[data_idx]['images'][idx]
                if img_info['subject'] == curr_data['subject'] \
                    and img_info['action_idx'] == curr_data['action_idx'] \
                    and img_info['subaction_idx'] == curr_data['subaction_idx'] \
                    and img_info['cam_idx'] == curr_data['cam_idx']:
                    frames.append(img_info)
                    last_data = img_info
                else:
                    frames.append(last_data)

        return frames

    def __getitem__(self, idx):
        data_idx = 0
        total = 0
        for i in range(len(self.all_counts)):
            count = self.all_counts[i]
            if idx < total + count:
                data_idx = i
                break
            total += count

        img_info = self.all_data[data_idx]['images'][idx - total]
        ann_info = self.all_data[data_idx]['annotations'][idx - total]
        # print(img_info)

        subject = img_info['subject']
        act_idx = img_info['action_idx']
        subact_idx = img_info['subaction_idx']
        cam_idx = img_info['cam_idx']
        frame_idx = img_info['frame_idx']
        width = img_info['width']
        height = img_info['height']

        extra_frames = self._load_extra_frames(img_info, data_idx, idx - total, 2)

        out_w = 1024
        out_h = 1024

        pad = []
        pad.append((out_w - width) // 2)
        pad.append(out_w - width - pad[0])
        pad.append( (out_h - height) // 2 )
        pad.append(out_h - height - pad[2])

        # sub_folder = f"s_{subject:02d}_act_{act_idx:02d}_subact_{subact_idx:02d}_ca_{cam_idx:02d}"
        # print("GetFrame", sub_folder, " match", img_info['file_name'])

        rot, trans, c, f = self._load_camera_params(subject, cam_idx)

        c[0] = c[0] + pad[0]
        c[1] = c[1] + pad[2]


        offset = idx - total
        ck = -1
        jk = -1
        current_size = 0
        for ak, av in self.all_joints[data_idx].items():
            for sk, sv in av.items():
                # print("##", ak, type(ak), act_idx, type(act_idx), sk, subact_idx, len(sv))
                if int(ak) == act_idx and int(sk) == subact_idx:
                    current_size = len(sv)
                    ck = offset // current_size
                    jk = offset % current_size
                    break
                offset -= len(sv) * 4

        # print("ck must be cam_idx", ck + 1, cam_idx, " and JK", jk)

        assert ck + 1 == cam_idx

        frame_info = self.all_joints[data_idx][f"{act_idx}"][f"{subact_idx}"][f"{jk}"]

        # print(frame_info)

        joints_2d = []
        for xyz in frame_info:
            xyz = np.dot(rot, xyz - trans)
            x, y = CamProj(xyz[0], xyz[1], xyz[2], f[0], f[1], c[0], c[1])
            joints_2d.append([x, y, xyz[2], 1]) # x,y,z, flag=0 occluded, 1 visible, 2 out of pic


        joints_2d = self._adjust_joints_2d(joints_2d, ann_info['keypoints_vis'], out_h, out_w)


        full_fn = os.path.join(self._data_folder, "images", img_info['file_name'])
        t = T.ToTensor()(Image.open(full_fn))
        tensor = torch.nn.functional.pad(t, pad)

        extra_tensors = []
        for info in extra_frames:
            full_fn = os.path.join(self._data_folder, "images", info['file_name'])
            t = T.ToTensor()(Image.open(full_fn))
            t = torch.nn.functional.pad(t, pad)
            extra_tensors.append(t)

        sample = dict(
            tensor=tensor,
            extra=extra_tensors,
            R=rot,
            T=trans,
            C=c,
            F=f,
            joints=np.array(frame_info),
            joints_2d=np.array(joints_2d)
        )

        sample = self._transform(sample)

        _, h, w = sample['tensor'].shape
        sample['heatmap'] = create_heatmap(sample['joints_2d'], h, w, 8, 7)

        sample['paf'] = create_paf_map(sample['joints_2d'], h, w, 8, 1)

        # print("extra count", len(sample['extra']))

        sample['extra'] = torch.cat(sample['extra'], 0)

        return sample

    def _adjust_joints_2d(self, joints, visibles, h, w):
        for pp, v in zip(joints, visibles):
            if v is not True:
                pp[3] = 0

            if pp[0] == 0 and pp[1] == 0:
                pp[3] = 2

            if pp[0] < 0 or pp[0] >= w or pp[1] < 0 or pp[1] >= h:
                pp[3] = 2

        return joints

    def __getitem__2(self, idx):
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(os.path.join(self._images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        mask = np.ones(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        # mask = get_mask(label['segmentations'], mask)
        sample = {
            'label': label,
            'image': image,
            'mask': mask
        }
        if self._transform:
            sample = self._transform(sample)

        mask = cv2.resize(sample['mask'], dsize=None, fx=1/self._stride, fy=1/self._stride, interpolation=cv2.INTER_AREA)
        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)
        for idx in range(keypoint_mask.shape[0]):
            keypoint_mask[idx] = mask
        sample['keypoint_mask'] = keypoint_mask

        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        for idx in range(paf_mask.shape[0]):
            paf_mask[idx] = mask
        sample['paf_mask'] = paf_mask

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))
        return sample

    def __len__(self):
        return sum(self.all_counts)

if __name__ == '__main__':
    print("human3.6m datasets annon test, subject 01")
    images_path = "D:/datasets/human3.6m/images"
    data_fn = "D:/datasets/human3.6m/annotations/annotations/Human36M_subject1_data.json"
    joint_fn = "D:/datasets/human3.6m/annotations/annotations/Human36M_subject1_joint_3d.json"
    camera_fn = "D:/datasets/human3.6m/annotations/annotations/Human36M_subject1_camera.json"

    img_fn = ""
    with open(data_fn, "rb") as fp:
        dd = json.load(fp)
        print(dd.keys())
        print("count of images:", len(dd['images']))
        print("count of annotations:", len(dd['annotations']))

        print("image format ", dd['images'][0])
        print("annotation format ", dd['annotations'][0])
        img_fn = dd['images'][0]['file_name']


    x, y, z = [0] * 3
    points = []
    with open(joint_fn, 'rb') as fp:
        dd = json.load(fp)
        print("list of actor id", dd.keys())
        print("count of actors", len(dd))
        print("list of subact", dd['2'].keys(), len(dd['2']['1']))
        # print(len(dd['2']['1']['0']))
        # print("action 2 subaction 1, keys ", dd['2']['1'].keys())
        print("frame 0 of actor 2, subact 1: ", dd['2']['1']['0'], len(dd['2']['1']['0']))
        x, y, z = dd['2']['1']['0'][0]

        points = dd['2']['1']['0']

        total = 0
        for dk in dd.keys():
            actions = dd[dk]
            for sk in actions.keys():
                jj = actions[sk]
                total += len(jj)
                print("actor", dk, "subact", sk, "count", len(jj))

        print("total joints frames", total)

    import numpy as np
    f = np.zeros((2), dtype=np.float32)
    c = np.zeros((2), dtype=np.float32)
    rot = np.zeros((3, 3), dtype=np.float32)
    trans = np.zeros((3), dtype=np.float32)

    with open(camera_fn, 'rb') as fp:
        dd = json.load(fp)
        print("Camera 01", dd['1'])
        f = np.array(dd['1']['f'])
        c = np.array(dd['1']['c'])
        rot = np.array(dd['1']['R'])
        # rot = np.transpose(rot)
        trans = np.array(dd['1']['t'])

    print("R", rot)
    print("t", trans)
    print("f", f)
    print("c", c)

    xyz = np.array([x, y, z], dtype=np.float32)
    print("jt xyz", xyz)

    xyz = np.dot(rot, xyz - trans)

    print("convert xyz to image xy")
    print("xyz", x, y, z, " converted ", xyz)
    ix, iy = CamProj(xyz[0], xyz[1], xyz[2], f[0], f[1], c[0], c[1], 25.0)
    print("XY", ix, iy)

    import cv2
    from os.path import join
    im = cv2.imread(join(images_path, img_fn), cv2.IMREAD_COLOR)
    cv2.circle(im, (int(ix), int(iy)), 5, [0, 0, 255], -1)

    pad = []
    pad.append(150) # up
    pad.append(20) # down
    pad.append(80) # left
    pad.append(16) # right
    padded_img = cv2.copyMakeBorder(im, pad[0], pad[1], pad[2], pad[3],
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # load camera from new json
    cam_fn = "D:/datasets/human3.6m/annotations/h36m/cameras.json"
    with open(cam_fn, "rb") as fp:
        cams = json.load(fp)
        rot = np.array(cams['subject1']['camera1']['R']).transpose()
        trans = np.array(cams['subject1']['camera1']['T']).reshape((3))
        c = np.array(cams['subject1']['camera1']['c']).reshape((2))
        c[0] = c[0] + pad[2]
        c[1] = c[1] + pad[0]
        f = np.array(cams['subject1']['camera1']['f']).reshape((2))

        fp.close()


    print("T", trans)

    cc = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

    for i in range(len(points)):
        # if i != 10:
        #     continue
        pp = points[i]
        xyz = np.array(pp, dtype=np.float32)
        print(rot.shape, trans.shape, c.shape, f.shape, xyz.shape)
        print(xyz, trans, xyz - trans)

        xyz = np.dot(rot, xyz - trans)
        ix, iy = CamProj(xyz[0], xyz[1], xyz[2], f[0], f[1], c[0], c[1], 1)
        print(xyz, " to ", ix, iy)
        cc = [0, 0, 255]
        if i == 7:
            cc = [0, 255, 0]
        elif i == 10:
            cc = [255, 255, 255]
        cv2.circle(padded_img, (int(ix), int(iy)), 3, cc, -1)


    # orig_jt_fn = "D:\\datasets\\human3.6m\\annotations\\h3.6m\\dataset\\S1\\directions_1.txt"
    # with open(orig_jt_fn, "rb") as fp:
    #     for line in fp:
    #         line = line.strip()
    #         pp = line.decode('utf-8').split(",")
    #         print("Point count", len(pp) / 3)
    #         print(pp)
    #         break

    cv2.imshow("test", padded_img)

    key = cv2.waitKey(0)
    if key == 27:  # esc
        exit(0)

