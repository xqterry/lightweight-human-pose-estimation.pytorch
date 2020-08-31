import torch
from models.with_resnet34 import PoseEstimationWithResnet50Single, ResNetFeatures, Bottleneck

if __name__ == '__main__':
    # ff = ResNetFeatures(Bottleneck, [3, 4, 6, 3])
    # print(ff.layer2)
    model = PoseEstimationWithResnet34Single()
    # print(net.backbone.layer2)

    x = torch.randn(1, 3, 256, 256)
    # f = ff(x)
    # print(f.shape)

    y = model(x)
    print(y.shape)

    from torchvision.models import resnet50

    print("load pretrained weights")
    net = resnet50(True)

    st = net.state_dict()
    dt = model.state_dict()
    # print(st.keys(), dt.keys())
    for k, v in dt.items():
        if k.find('ghost') != -1 or k.find('head') != -1 or k.find('backbone') != -1:
            sk = k[k.find('.') + 1:]
            # print("Find k ", sk, " by ", k)
            if dt[k].shape != st[sk].shape:
                print("unmatch", k)
            dt[k] = st[sk]

    model.load_state_dict(dt)
    torch.save(model.state_dict(), "../resnet50_single_checkpoints/init_r50_checkpoints.pt")