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
from models.with_mobilenet import PoseEstimationWithMobileNetMultiple

from shutil import copyfile

from augment_test import rand_motion_blur_weight

from PIL import Image

def masked_l2_loss(output, label, mask, batch):
    loss = (output - label) * mask
    loss = (loss * loss) / 2 / batch

    return loss.sum()

from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA

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
    import datetime
    st = time.time()

    batch = 40
    batches_per_iter = 1
    log_after = 3
    checkpoint_after = 600
    num_refinement_stages = 3
    eval_after = 13

    dataset = HM36Dataset("d:/datasets/human3.6m", 8, 7, 1, train_list=None,
                          transform=T.Compose([
                                HM2DScale(),
                                HM2DRotate(pad=(0.5, 0.5, 0.5)),
                                HM2DCropPad(crop_x=368, crop_y=368),
                                HM2DFlip(0.5),
                            ]),
                          paf_ver=2,
                          vgg_norm=False
                          )
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)

    from DGPT.Visualize.Viz import Viz

    viz = Viz(dict(env="mob_mf"))

    print("load data cost ", time.time() - st)

    # net = PoseEstimationWithMobileNetMultiple(num_refinement_stages=num_refinement_stages)
    net = PoseEstimationWithMobileNetMultiple(3, 128, 18, 32)

    base_lr = 4e-4

    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.ghost0, 'weight')},
        {'params': get_parameters_conv(net.ghost1, 'weight')},
        {'params': get_parameters_conv(net.ghost2, 'weight')},
        {'params': get_parameters_conv(net.phase0, 'weight')},
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.phase0, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.ghost0, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.ghost1, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.ghost2, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.phase0, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.ghost0, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.ghost1, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.ghost2, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.phase0, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_bn(net.ghost0, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_bn(net.ghost1, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_bn(net.ghost2, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},

        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)

    num_iter = 0
    current_epoch = 0
    drop_after_epoch = [100, 200, 260]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)

    criterion = torch.nn.MSELoss()

    checkpoints_folder = "./multiframe_checkpoints"
    checkpoint_fn = "./multiframe_checkpoints/MOBv3.pth"
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
        print('load from init checkpoints')
        net.load_state_dict(torch.load("./multiframe_checkpoints/init_mobile_mf_checkpoints.pt"))

    net = net.cuda()
    net.train()

    heatmap_mask = torch.ones(batch, 18, 46, 46).cuda()

    paf_masks = 1

    start_time = time.time()

    for epochId in range(current_epoch, 280):
        total_losses = [0, 0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            # print(type(batch_data['tensor']), type(batch_data['extra']))
            # print(batch_data['tensor'].shape, batch_data['extra'].shape)
            # print(batch_data['heatmap'].shape, batch_data['paf'].shape)

            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            x = batch_data['tensor'].cuda()

            x1 = batch_data['extra'][:, :3, ...].cuda()
            x2 = batch_data['extra'][:, 3:, ...].cuda()

            # print(x.min(), x.max()) # to bgr ?
            x = x[:, [2, 1, 0], ...]
            x1 = x1[:, [2, 1, 0], ...]
            x2 = x2[:, [2, 1, 0], ...]

            x -= 0.5
            x1 -= 0.5
            x2 -= 0.5

            # x = preprocess(x)
            # x1 = preprocess(x1)
            # x2 = preprocess(x2)

            # ww = rand_motion_blur_weight(3, 10, 360).cuda()
            # x = torch.nn.functional.conv2d(x, ww)
            # x1 = torch.nn.functional.conv2d(x1, ww)
            # x2 = torch.nn.functional.conv2d(x2, ww)

            heatmap = torch.cat([batch_data['heatmap'][:, :17, ...], batch_data['heatmap'][:, 255:, ...]], 1)
            heatmap = heatmap.cuda()
            paf = batch_data['paf'].cuda()

            stages_output = net(x, x1, x2)

            losses = []
            for loss_idx in range(len(total_losses) // 2):
                loss0 = masked_l2_loss(stages_output[loss_idx * 2], heatmap, heatmap_mask, batch)
                loss1 = masked_l2_loss(stages_output[loss_idx * 2 + 1], paf, paf_masks, x.shape[0])

                losses.append(loss0)
                losses.append(loss1)
                total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            loss /= batches_per_iter
            loss.backward()

            viz.draw_line(num_iter, loss.item(), "Loss")

            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1

                scheduler.step()
            else:
                continue

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses) // 2):
                    print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                        loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                        loss_idx + 1, total_losses[loss_idx * 2] / log_after))
                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0

                xx = x[:1, ...].detach()
                hh = heatmap[:1, :17, ...].detach()

                hh = hh.squeeze(0).reshape(17, 1, hh.shape[2], hh.shape[3])

                print(xx.shape, hh.shape)
                viz.draw_images(xx, "inputs4")
                viz.draw_images(hh, "inputs_heatmap4")

                print("cost ", time.time() - start_time, " per ", log_after, "@", str(datetime.datetime.now()))
                start_time = time.time()
            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/mob_mf_v3_checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
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
                t = torch.nn.functional.interpolate(t, (368, 368), mode='bilinear', align_corners=False)

                blur = GaussianBlur_CUDA(0.5)
                # t = blur(t)

                t1 = t.clone()
                t2 = t.clone()

                outputs = net(t, t1, t2)

                heatmaps = outputs[-2][:, :17]  # .squeeze(0)
                paf_maps = outputs[-1]  # .squeeze(0)

                heatmap2 = heatmaps.sum(1, keepdim=True)
                heatmap2 = torch.nn.functional.interpolate(heatmap2, size=512, mode='bilinear', align_corners=False)
                # heatmap = heatmap[:, :, :255].sum(2)
                heatmap = heatmaps.squeeze(0).reshape(17, 1, heatmaps.shape[2], heatmaps.shape[3])
                viz.draw_images(heatmap, "output_heatmap")
                viz.draw_images(heatmap2, "sum_heatmap")

                net.train()
