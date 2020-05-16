"""A U-net model with clearer code."""

import torch
from torch import nn
from torch.nn import functional as F

from src.train_testers import GraspTrainTester


class DoubleConv(nn.Module):
    """Two conv. layers with batch norm."""

    def __init__(self, in_channels, out_channels):
        """Initialize_layers."""
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature):
        """Forward pass."""
        return self.double_conv(feature)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        """Initialize_layers."""
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, feature):
        """Forward pass."""
        return self.maxpool_conv(feature)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels):
        """Initialize_layers."""
        super().__init__()
        self.upscale = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, new, old):
        """Forward pass."""
        new = self.upscale(new)
        diff_y = old.shape[2] - new.shape[2]
        diff_x = old.shape[3] - new.shape[3]
        half_y = diff_y // 2
        half_x = diff_x // 2
        new = F.pad(new, (half_x, diff_x - half_x, half_y, diff_y - half_y))
        return self.conv(torch.cat([old, new], dim=1))


class UNet(nn.Module):
    """Original U-net."""

    def __init__(self, input_channels=1, output_channels=1):
        """Initialize_layers."""
        super().__init__()
        self.num_outs = output_channels

        # Encoders
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        # Decoders
        self.pos_decoder = nn.Conv2d(64, output_channels, 1)
        self.cos_decoder = nn.Conv2d(64, output_channels, 1)
        self.sin_decoder = nn.Conv2d(64, output_channels, 1)
        self.width_decoder = nn.Conv2d(64, output_channels, 1)
        self.graspness_decoder = nn.Conv2d(64, 1, 1)
        self.bin_classifier = nn.Conv2d(64, output_channels, 1)

    def forward(self, img):
        """Forward pass."""
        out0 = self.inc(img)
        out1 = self.down1(out0)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        img = self.up1(out4, out3)
        img = self.up2(img, out2)
        img = self.up3(img, out1)
        img = self.up4(img, out0)
        return (
            self.pos_decoder(img).squeeze(1),
            self.cos_decoder(img).squeeze(1),
            self.sin_decoder(img).squeeze(1),
            self.width_decoder(img).squeeze(1),
            self.graspness_decoder(img).squeeze(1),
            self.bin_classifier(img).squeeze(1)
        )


def train_test(config, model_params={}):
    """Train and test a net."""
    net = UNet(4 if config.use_rgbd_img else 1, config.num_of_bins)
    features = {'depth_images', 'grasp_targets', 'grasps'}
    train_tester = GraspTrainTester(net, config, features)
    train_tester.train_test()
