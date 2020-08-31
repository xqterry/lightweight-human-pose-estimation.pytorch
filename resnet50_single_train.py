import os
import torch
from torch.utils.data import DataLoader

from datasets.human36m import HM36Dataset, HM2DCropPad, HM2DRotate, HM2DFlip, HM2DScale

import time
import numpy as np
import cv2

import torch.optim as optim
import torchvision.transforms as T

from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_resnet34 import PoseEstimationWithResnet50Single

from shutil import copyfile

from augment_test import rand_motion_blur_weight

from PIL import Image

def masked_l2_loss(output, label, mask, batch):
    loss = (output - label) * mask
    loss = (loss * loss) / 2 / batch

    return loss.sum()

class JointsMSELoss(torch.nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA
from DGPT.Visualize.Viz import Viz

def preprocess(tensor):
    g = np.random.randint(0, 7)
    n = np.random.sample() * 0.01
    tensor = tensor + n
    if g > 0:
        degrade_blur = GaussianBlur_CUDA(g / 18)
        tensor = degrade_blur(tensor)

    pre_blur = GaussianBlur_CUDA(1)
    tensor = pre_blur(tensor)

    if np.random.sample() < 0.5:
        ww = rand_motion_blur_weight(3, 5, 360).cuda()
        tensor = torch.nn.functional.conv2d(tensor, ww, None, 1, ww.shape[2] // 2)

    return tensor

if __name__ == '__main__':
    st = time.time()

    viz = Viz(dict(env="res50single"))
    batch = 64
    batches_per_iter = 1
    log_after = 10
    checkpoint_after = 100
    eval_after = 13
    num_refinement_stages = 3

    dataset = HM36Dataset("d:/datasets/human3.6m", 4, 7, 1, train_list=None,
                          transform=T.Compose([
                                HM2DScale(),
                                HM2DRotate(pad=(0.5, 0.5, 0.5)),
                                HM2DCropPad(crop_x=256, crop_y=256),
                                HM2DFlip(0.5),
                            ])
                          )
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=1)

    print("load data cost ", time.time() - st)

    net = PoseEstimationWithResnet50Single()

    base_lr = 1e-3

    optimizer = optim.Adam(net.parameters(), lr=base_lr)

    num_iter = 0
    current_epoch = 0
    drop_after_epoch = [100, 200, 260]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)

    checkpoints_folder = "./resnet50_single_checkpoints"
    checkpoint_fn = os.path.join(checkpoints_folder, "S.pth")
    if os.path.exists(checkpoint_fn):
        checkpoint = torch.load(checkpoint_fn)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        scheduler.load_state_dict(checkpoint['scheduler'])
        num_iter = checkpoint['iter']
        current_epoch = checkpoint['current_epoch']

        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(torch.load(os.path.join(checkpoints_folder, "init_r50_checkpoints.pt")))

    net = net.cuda()
    net.train()

    heatmap_mask = torch.ones(batch, 17, 64, 64).cuda()

    paf_masks = 1

    criterion = JointsMSELoss(
        use_target_weight=False
    ).cuda()

    for epochId in range(current_epoch, 280):
        total_losses = 0
        batch_per_iter_idx = 0
        start_time = time.time()
        for batch_data in train_loader:
            # print(type(batch_data['tensor']), type(batch_data['extra']))
            # print(batch_data['tensor'].shape, batch_data['extra'].shape)
            # print(batch_data['heatmap'].shape, batch_data['paf'].shape)

            if batch_per_iter_idx == 0:
                optimizer.zero_grad()


            x = batch_data['tensor'].cuda()
            # x1 = batch_data['extra'][:, :3, ...].cuda()
            # x2 = batch_data['extra'][:, 3:, ...].cuda()

            # x = preprocess(x)
            # x1 = preprocess(x1)
            # x2 = preprocess(x2)

            # ww = rand_motion_blur_weight(3, 10, 360).cuda()
            # x = torch.nn.functional.conv2d(x, ww)
            # x1 = torch.nn.functional.conv2d(x1, ww)
            # x2 = torch.nn.functional.conv2d(x2, ww)

            heatmap = batch_data['heatmap'].cuda()
            paf = batch_data['paf'].cuda()

            stages_output = net(x)

            # print(stages_output.shape, heatmap.shape)

            loss = criterion(stages_output, heatmap[:, :17, ...], None)

            loss.backward()

            viz.draw_line(num_iter, loss.item(), "Loss")

            total_losses += loss.item() / batches_per_iter
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1

                scheduler.step()
            else:
                continue

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter), "cost ", time.time() - start_time)
                print("Loss: ", total_losses / log_after)
                total_losses = 0
                start_time = time.time()

                show_maps = heatmap[:4, :17, ...]
                show_maps = show_maps.sum(1, keepdim=True)
                viz.draw_images(show_maps, "log_heatmap")

            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                torch.save({'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name)

                copyfile(snapshot_name, checkpoint_fn)

            if num_iter % eval_after == 0:
                net.eval()

                input_fn = "D:/datasets/coco/val2017/000000000785.jpg"
                # input_fn = "D:/datasets/mh.jpg"
                # input_fn = "D:/datasets/coco/val2017/000000000776.jpg"
                # input_fn = "D:/datasets/human3.6m/images/s_11_act_16_subact_02_ca_03/s_11_act_16_subact_02_ca_03_000003.jpg"

                t = T.ToTensor()(Image.open(input_fn))

                _, oh, ow = t.shape
                sz = max(oh, ow)
                # print("input size", ow, oh, sz)

                pad = []
                pad.append((sz - ow) // 2)
                pad.append(sz - ow - pad[0])
                pad.append((sz - oh) // 2)
                pad.append(sz - oh - pad[2])

                t = torch.nn.functional.pad(t, pad, mode='constant', value=0.5)

                t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)

                t = t.cuda().unsqueeze(0)
                t = torch.nn.functional.interpolate(t, (256, 256), mode='bilinear', align_corners=False)

                blur = GaussianBlur_CUDA(0.5)
                # t = blur(t)

                # t1 = t.clone()
                # t2 = t.clone()

                heatmaps = net(t)

                heatmap = heatmaps.sum(1, keepdim=True)
                # heatmap = heatmap[:, :, :255].sum(2)
                viz.draw_images(heatmap, "output_heatmap")

                net.train()
