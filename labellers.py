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
    def forward(self, wav):
        """Returns the label for the waveform autodecoder.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        return wav.unsqueeze(2)


class LPSLabeller(torch.nn.Module):
    def forward(self, compute_STFT,wav):
        feats = compute_STFT(wav)
        feats = spectral_magnitude(feats, power=1)# power spectrum

        # Log1p reduces the emphasis on small differences
        feats = torch.log1p(feats)

        return feats

class LIMLabeller(torch.nn.Module):
    def forward(self, pred):
        bsz, slen = pred.size(0) // 2, pred.size(1)

        return torch.cat((
            torch.ones(bsz, slen, 1, requires_grad=False),
            torch.zeros(bsz, slen, 1, requires_grad=False)
            ),
            dim=0,
        )

class GIMLabeller(torch.nn.Module):
    def forward(self, pred):
        bsz, slen = pred.size(0) // 2, pred.size(1)

        return torch.cat((
            torch.ones(bsz, slen, 1, requires_grad=False),
            torch.zeros(bsz, slen, 1, requires_grad=False)
            ),
            dim=0,
        )

class SPCLabeller(torch.nn.Module):
    def forward(self, pred):
        bsz, slen = pred.size(0) // 2, pred.size(1)

        return torch.cat((
            torch.ones(bsz, slen, 1, requires_grad=False),
            torch.zeros(bsz, slen, 1, requires_grad=False)
            ),
            dim=0,
        )
