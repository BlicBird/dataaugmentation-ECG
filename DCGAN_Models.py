"""
Discriminator and Generator sub-models for DCGAN
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, ecg_length, embed_channels):
        super(Discriminator, self).__init__()
        self.ecg_length = ecg_length
        self.embed_channels = embed_channels
        self.disc = nn.Sequential(
            # input: 32 x Channels x ECG length
            nn.Conv1d(
                channels_img + embed_channels, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
        self.embed = nn.Embedding(num_classes, ecg_length)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], self.embed_channels, self.ecg_length)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, noise_dim, embed_channels):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embed_channels = embed_channels
        self.net = nn.Sequential(
            # Input: 32 x Channels x Noise_len
            self._block(channels_noise + embed_channels, features_g * 16, 4, 1, 1),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose1d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: 32 x Channels x ECG length
        )

        self.embed = nn.Embedding(num_classes, noise_dim)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], self.embed_channels, self.noise_dim)
        x = torch.cat([x,embedding], dim=1)
        return self.net(x)


def initialize_weights(model):
    
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)