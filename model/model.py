import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class IrisBlock(nn.Module):
    """This is the main building block for architecture"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(IrisBlock, self).__init__()

        # My impl
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        padding = (kernel_size - 1) // 2
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.convAct = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels / 2), kernel_size=stride, stride=stride,
                      padding=0, bias=True),
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.PReLU(int(out_channels / 2)),
        )
        self.dwConvConv = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels / 2), out_channels=int(out_channels / 2),
                      kernel_size=kernel_size, stride=1, padding=padding,  # Padding might be wrong here
                      groups=int(out_channels / 2), bias=True),
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.BatchNorm2d(out_channels),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        h = self.convAct(x)
        if self.stride == 2:
            x = self.max_pool(x)

        h = self.dwConvConv(h)

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(h + x)


class IrisLandmarks(nn.Module):
    """The IrisLandmark face landmark model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.
    """

    def __init__(self):
        super(IrisLandmarks, self).__init__()

        # self.num_coords = 228
        # self.x_scale = 64.0
        # self.y_scale = 64.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2)
        )
        self.split_eye = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=2, stride=1, padding=0, bias=True)
        )
        self.split_iris = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=2, stride=1, padding=0, bias=True)
        )
        self.split_pupil = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=2, stride=1, padding=0, bias=True)
        )
        self.gaze_prediction = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        self.new_backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

        )


        self.resnet18 = torchvision.models.resnet18(pretrained=False, num_classes=2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         pass
        #         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         # nn.init.xavier_normal_(m.weight, gain=1)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         pass
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        # x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        # b = x.shape[0]  # batch size, needed for reshaping later
        #
        # x = self.new_backbone(x)  # (b, 128, 8, 8)

        # gaze_feature = self.gaze_prediction(x)
        #
        # gaze_feature = gaze_feature.view(gaze_feature.size(0), -1)
        #
        # gaze_data = self.fc(gaze_feature)

        # e = self.split_eye(x)  # (b, 68, 1, 1)
        # e = e.view(b, -1)  # (b, 68)
        #
        # i = self.split_iris(x)  # (b, 16, 1, 1)
        # i = i.reshape(b, -1)  # (b, 16)"""

        # p = self.split_pupil(x)  # (b, 16, 1, 1)
        # p = p.reshape(b, -1)  # (b, 16, 1, 1)"""
        x = self.resnet18(x)
        p = x.view(x.size(0), -1)
        # x = self.new_backbone(x)
        # x = x.view(x.size(0), -1)
        # p = self.fc(x)

        return p.view(-1, 2)


