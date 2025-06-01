import torch
import torch.nn as nn


import torch.nn.functional as F


class DCGANGenerator(nn.Module):
    def __init__(self, inputDim=100, outputChannels=3):
        super(DCGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(inputDim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, outputChannels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.model(input)
        # Center crop from 32x32 to 28x28
        return F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)


class DCGANDiscriminator(nn.Module):
    def __init__(self, inputChannels=3):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 28 x 28
            nn.Conv2d(inputChannels, 32, 4, 2, 1,
                      bias=False),   # -> 32 x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),

            # -> 64 x 7 x 7
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # -> 128 x 4 x 4
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # -> 256 x 2 x 2
            nn.Conv2d(128, 256, 3, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 1, 1, 0, bias=False),              # -> 1 x 1 x 1
            # nn.Sigmoid()  # Optional: use for non-WGAN variants
        )

    def forward(self, input):
        return self.main(input).view(-1)

# taken from: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generator
# Custom weights initialization


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
