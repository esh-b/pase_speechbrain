import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d, SincConv
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.lobes.models.dual_path import Decoder


class PASEEncoder(torch.nn.Module):
    def __init__(
        self,
        device='cpu',
        activation=torch.nn.PReLU,
        use_sincnet=True,
        blocks_channels=[64,128,128,256,256,512,512,100],
        blocks_kernel_sizes=[251,20,11,11,11,11,11,11,1],
        blocks_strides=[1,10,2,1,2,1,2,2,1],
        in_channels=1,

    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        first_hidden_layer = SincConv if use_sincnet else Conv1d
        self.blocks.append(
            first_hidden_layer(
                in_channels=in_channels,
                out_channels=blocks_channels[0],
                kernel_size=blocks_kernel_sizes[0],
                stride=blocks_strides[0],
            )
        )
        in_channels = sincnet_channels
        cnn_blocks = len(blocks_channels) - 1

        for block_index in range(cnn_blocks):
            out_channels = cnn_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=cnn_kernel_sizes[block_index],
                        stride=cnn_strides[block_index],
                    ),
                    activation(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = cnn_channels[block_index]

    def forward(self, x, *args, **kwargs):
        for layer in self.blocks:
            try:
                x = layer(x, *args, **kwargs)
            except TypeError:
                x = layer(x)
        return x
