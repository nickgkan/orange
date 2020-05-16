"""A GGCNN2 model with clearer code."""

from torch import nn

from src.train_testers import GraspTrainTester


class GGCNN2(nn.Module):
    """Original GGCNN2."""

    def __init__(self, input_channels=1, output_channels=1):
        """Initialize layers."""
        super().__init__()
        self.num_outs = output_channels
        sizes = [16, 16, 32, 16]  # intermediate filter sizes
        self.enc_size = sizes[-1]

        # Encoder
        self.encoder = nn.Sequential(
            # 1st DoubleConv filter (without batch norm)
            nn.Conv2d(input_channels, sizes[0], 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[0], sizes[0], 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd DoubleConv filter
            nn.Conv2d(sizes[0], sizes[1], 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[1], sizes[1], 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions (padding = 2 * dilation)
            nn.Conv2d(sizes[1], sizes[2], 5, dilation=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[2], sizes[2], 5, dilation=4, padding=8),
            nn.ReLU(inplace=True),

            # Transpose convolutions
            nn.ConvTranspose2d(sizes[2], sizes[3], 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(sizes[3], sizes[3], 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoders
        self.pos_decoder = nn.Conv2d(sizes[3], output_channels, 1)
        self.cos_decoder = nn.Conv2d(sizes[3], output_channels, 1)
        self.sin_decoder = nn.Conv2d(sizes[3], output_channels, 1)
        self.width_decoder = nn.Conv2d(sizes[3], output_channels, 1)
        self.graspness_decoder = nn.Conv2d(sizes[3], 1, 1)
        self.bin_classifier = nn.Conv2d(sizes[3], output_channels, 1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight, gain=1)

    def forward(self, img):
        """Forward pass, the input is a(n) (depth) image tensor."""
        img = self.encoder(img)
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
    net = GGCNN2(4 if config.use_rgbd_img else 1, config.num_of_bins)
    features = {'depth_images', 'grasp_targets', 'grasps'}
    train_tester = GraspTrainTester(net, config, features)
    train_tester.train_test()
