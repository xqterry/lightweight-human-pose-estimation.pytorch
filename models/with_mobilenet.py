import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output


class PoseEstimationWithMobileNetMultiple(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.ghost0 = conv(     3,  32, stride=2, bias=False)
        self.ghost1 = conv(     3,  32, stride=2, bias=False)
        self.ghost2 = conv(     3,  32, stride=2, bias=False)

        self.phase0 = conv_dw( 32,  64)

        # self.phase1 = nn.Sequential(
        #     conv_dw(32, 64),
        # )

        self.model = nn.Sequential(
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x, x1, x2):
        x = self.ghost0(x)
        x1 = self.ghost1(x1)
        x2 = self.ghost2(x2)
        x3 = torch.cat([x, x1 + x2], 1) # c 64

        x = self.phase0(x)

        backbone_features = self.model(x + x3)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output

if __name__ == '__main__':
    model = PoseEstimationWithMobileNetMultiple(3, 128, 18, 32)
    # print(net)
    x = torch.randn(1, 3, 368, 368)
    x1 = x.clone()
    x2 = x.clone()

    oo = model(x, x1, x2)
    print(oo[-2].shape)

    net = PoseEstimationWithMobileNet(3)
    net.load_state_dict(torch.load("../checkpoint_iter_370000.pth")['state_dict'])

    st = net.state_dict()
    dt = model.state_dict()
    print(st.keys())
    print(dt.keys())
    for k, v in dt.items():
        if k.find('ghost') != -1:# or k.find('phase') != -1 or k.find('model') != -1:
            sk = k[k.find('.') + 1:]
            sk = f"model.0.{sk}"
            # print("GHOST Find k ", sk, " by ", k)
            if dt[k].shape != st[sk].shape:
                print("unmatch", k)
            dt[k] = st[sk]

        elif k.find('phase') != -1:
            sk = k[k.find('.') + 1:]
            sk = f"model.1.{sk}"
            # print("PHASE Find k ", sk, " by ", k)
            if dt[k].shape != st[sk].shape:
                print("unmatch", k)
            dt[k] = st[sk]

        elif k.find('model') != -1:
            p0 = k[:k.find('.')]
            p1 = k[k.find('.') + 1:]
            idx = p1[:p1.find('.')]
            p2 = p1[p1.find('.')+1:]

            idx = int(idx)
            print(k, " IDX ", idx)
            sk = f"{p0}.{idx + 2}.{p2}"
            print("MODEL Find k ", sk, " by ", k)
            if dt[k].shape != st[sk].shape:
                print("unmatch", k)
            dt[k] = st[sk]
        else:
            if dt[k].shape != st[k].shape:
                print("OHTER ", k, dt[k].shape, st[k].shape)
                if dt[k].shape[0] < st[k].shape[0]:
                    dt[k] = st[k][:dt[k].shape[0], ...]
                elif dt[k].shape[1] < st[k].shape[1]:
                    dt[k] = st[k][:, :dt[k].shape[1], ...]
                else:
                    print("WTF", k)
            else:
                dt[k] = st[k]

    model.load_state_dict(dt)
    torch.save(model.state_dict(), "../multiframe_checkpoints/init_mobile_mf_checkpoints.pt")

