import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    This class defines the residual block, which forms the basis of the res-net architecture.
    Each block contains two convolutional layers and an activation function.
    Batch Normalization is used after the first convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, dropout_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class UpscaleBlock(nn.Module):
    """
    This class defines the upscale block, which is used to upscale the feature maps back to the input resolution.
    The block contains a convolutional layer followed by a PixelShuffle operation.
    """
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(UpscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class GlobalPropagation(nn.Module):
    """
    This class defines the overall network architecture for global propagation.
    The architecture consists of an input layer, a series of residual blocks, a series of upscale blocks, and an output layer.
    """
    def __init__(self, in_channels, out_channels, num_blocks, upscale_factor=2, use_batch_norm=True, dropout_prob=0.0):
        super(GlobalPropagation, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = self._make_layer(ResidualBlock, out_channels, out_channels, num_blocks, use_batch_norm, dropout_prob)
        self.upscale_blocks = self._make_layer(UpscaleBlock, out_channels, out_channels, num_blocks // 2, upscale_factor)
        self.output_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, *args):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels, *args))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        res_out = self.res_blocks(x)
        upscale_out = self.upscale_blocks(res_out)
        out = self.output_layer(upscale_out)
        return out
