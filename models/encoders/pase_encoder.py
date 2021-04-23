import torch
import torch.nn as nn

from torchqrnn import QRNN
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d, SincConv
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.lobes.models.dual_path import Decoder


class PASEEncoder(torch.nn.Module):
    def __init__(
        self,
        activation=torch.nn.PReLU,
        use_sincnet=True,
        use_qrnn=False,
        in_channels=1,
        blocks_channels=[64,64,128,128,256,256,512,512,100],
        blocks_kernel_sizes=[251,20,11,11,11,11,11,11,1],
        blocks_strides=[1,10,2,1,2,1,2,2,1],
        qrnn_hidden_neurons=512,
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
        in_channels = blocks_channels[0]

        # Convolution blocks (except last conv block)
        for block_index in range(1, len(blocks_channels)-1):
            out_channels = blocks_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=blocks_kernel_sizes[block_index],
                        stride=blocks_strides[block_index],
                    ),
                    BatchNorm1d(input_size=out_channels),
                    activation(),
                ]
            )
            in_channels = blocks_channels[block_index]

        # Add Q-RNN block
        if use_qrnn:
            self.blocks.append(
                QRNN(input_size=in_channels,
                     hidden_size=qrnn_hidden_neurons,
                     num_layers=1,
                     dropout=0,
                     window=2,
                )
            )
            in_channels = qrnn_hidden_neurons

        # Final convolution block
        self.blocks.extend(
            [
                Conv1d(
                    in_channels=in_channels,
                    out_channels=blocks_channels[block_index + 1],
                    kernel_size=blocks_kernel_sizes[block_index + 1],
                    stride=blocks_strides[block_index + 1],
                ),
                BatchNorm1d(input_size=blocks_channels[block_index + 1], affine=False),
            ]
        )

    def forward(self, x, *args, **kwargs):
        for layer in self.blocks:
            if layer._get_name() == 'QRNN':
                x = x.transpose(0, 1)
                x, _ = layer(x)
                x = x.transpose(0, 1)
            else:
                x = layer(x)
        return x
