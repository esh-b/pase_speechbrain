"""Basic feature pipelines.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import torch
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    Filterbank,
    DCT,
    Deltas,
    ContextWindow,
)


class DecoderLabeller(torch.nn.Module):
    """Generate labels for the waveform decoder worker.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 660])
    """

    def forward(self, wav):
        """Returns the label for the waveform autodecoder.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        return wav.unsqueeze(2)