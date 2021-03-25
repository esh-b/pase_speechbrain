import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvTranspose1d(nn.Module):
    """This class implements 1d transposed convolution with speechbrain dimension convention.
    Transpose convolution is normally used to perform upsampling.
    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        upsampling in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str or int
        if "valid", no padding is applied
        if "same", padding amount is inferred so that the output is closest to possible to input
        if an integer value is entered, a custom padding is used
    output_padding : int,
        Additional size added to one side of the output shape
    groups: int
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool
        If True, adds a learnable bias to the output
    skip_transpose : bool
        If True, uses batch x time x channel convention of speechbrain
    Example
    -------
    >>> inp_tensor = torch.rand([10, 12, 40]) #[batch, time, fea]
    >>> convtranspose_1d = ConvTranspose1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=3, stride=2
    ... )
    >>> out_tensor = convtranspose_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 25, 8])
    >>> # Combination of Conv1d and ConvTranspose1d
    >>> from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
    >>> signal = torch.tensor([1,100])
    >>> signal = torch.rand([1,100]) #[batch, time]
    >>> conv1d = Conv1d(input_shape=signal.shape, out_channels=1, kernel_size=3, stride=2)
    >>> conv_out = conv1d(signal)
    >>> conv_t = ConvTranspose1d(input_shape=conv_out.shape, out_channels=1, kernel_size=3, stride=2, padding=1)
    >>> signal_rec = conv_t(conv_out, output_size=[100])
    >>> signal_rec.shape
    torch.Size([1, 100])
    >>> signal = torch.rand([1,115]) #[batch, time]
    >>> conv_t = ConvTranspose1d(input_shape=signal.shape, out_channels=1, kernel_size=3, stride=2, padding='same')
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 115])
    >>> signal = torch.rand([1,115]) #[batch, time]
    >>> conv_t = ConvTranspose1d(input_shape=signal.shape, out_channels=1, kernel_size=3, stride=2, padding='valid')
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 231])
    >>> signal = torch.rand([1,115]) #[batch, time]
    >>> conv_t = ConvTranspose1d(input_shape=signal.shape, out_channels=1, kernel_size=3, stride=2, padding=10)
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 211])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        skip_transpose=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        if self.padding == "same":
            L_in = input_shape[-1] if skip_transpose else input_shape[1]
            padding_value = get_padding_elem_transposed(
                L_in,
                L_in,
                stride=stride,
                kernel_size=kernel_size,
                dilation=dilation,
                output_padding=output_padding,
            )
        elif self.padding == "valid":
            padding_value = 0
        elif type(self.padding) is int:
            padding_value = padding
        else:
            raise ValueError("Not supported padding type")

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding_value,
            groups=groups,
            bias=bias,
        )

    def forward(self, x, output_size=None):
        """Returns the output of the convolution.
        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """

        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        wx = self.conv(x, output_size=output_size)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        return in_channels


def get_padding_elem_transposed(
    L_out: int,
    L_in: int,
    stride: int,
    kernel_size: int,
    dilation: int,
    output_padding: int,
):
    """This function computes the required padding size for transposed convolution
    Arguments
    ---------
    L_out : int
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    output_padding : int
    """

    padding = -0.5 * (
        L_out
        - (L_in - 1) * stride
        - dilation * (kernel_size - 1)
        - output_padding
        - 1
    )
    return int(padding)
