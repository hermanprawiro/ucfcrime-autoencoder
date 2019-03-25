import torch
import torch.nn as nn

class Conv3DAE(nn.Module):
    def __init__(self, input_channels=3, base_size=32):
        super().__init__()
        self.input_channels = input_channels

        # Input shape (n, 3, 16, 224, 224)
        self.encoder = nn.Sequential(
            nn.BatchNorm3d(3),
            Conv3DBlock(self.input_channels, base_size, kernel_size=3, stride=(1, 2, 2), padding=1), # (n, base, 16, 112, 112)
            Conv3DBlock(base_size, base_size * 2, kernel_size=3, stride=2, padding=1), # (n, base * 2, 8, 56, 56)
            Conv3DBlock(base_size * 2, base_size * 4, kernel_size=3, stride=2, padding=1), # (n, base * 4, 4, 28, 28)
            Conv3DBlock(base_size * 4, base_size * 8, kernel_size=3, stride=2, padding=1), # (n, base * 8, 2, 14, 14)
            Conv3DBlock(base_size * 8, base_size * 16, kernel_size=3, stride=2, padding=1), # (n, base * 16, 1, 7, 7)
        )
        self.decoder = nn.Sequential(
            Conv3DBlock(base_size * 16, base_size * 8, kernel_size=4, stride=2, padding=1, deconv=True), # (n, base * 8, 2, 14, 14)
            Conv3DBlock(base_size * 8, base_size * 4, kernel_size=4, stride=2, padding=1, deconv=True), # (n, base * 4, 4, 28, 28)
            Conv3DBlock(base_size * 4, base_size * 2, kernel_size=4, stride=2, padding=1, deconv=True), # (n, base * 2, 8, 56, 56)
            Conv3DBlock(base_size * 2, base_size, kernel_size=4, stride=2, padding=1, deconv=True), # (n, base, 16, 112, 112)
            Conv3DBlock(base_size, 3, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, deconv=True, batchnorm=False, activation='sigmoid'), # (n, 3, 16, 224, 224)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu', batchnorm=True, deconv=False):
        super().__init__()

        if deconv:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        if batchnorm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)

        return x