import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.lobes.models.dual_path import Decoder


class WaveformWorker(torch.nn.Module):
    def __init__(
        self,
        device='cpu',
        decoder_blocks=3,
        decoder_channels=[100,100,100],
        decoder_kernel_sizes=[4,4,10],
        decoder_strides=[4,4,10],
        lin_neurons=256,
        in_channels,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(decoder_blocks):
            out_channels = decoder_channels[block_index]
            self.blocks.extend(
                [
                    Decoder(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=decoder_kernel_sizes[block_index],
                        stride=decoder_strides[block_index],
                    ),
                    activation(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = cnn_channels[block_index]

        self.blocks.append(
            Linear(n_neurons=lin_neurons),
        )


    def forward(self, x, *args, **kwargs):
        for layer in self.blocks:
            try:
                x = layer(x, *args, **kwargs)
            except TypeError:
                x = layer(x)
        return x
