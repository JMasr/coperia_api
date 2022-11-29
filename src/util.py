import json

import numpy as np
import requests
import torch
import torchaudio
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp

from src.config import Config


class FeatureExtractor:
    """ Class for feature extraction
    args: input arguments dictionary
    Mandatory arguments: resampling_rate, feature_type, window_size, hop_length
    For MFCC: f_max, n_mels, n_mfcc
    For MelSpec/logMelSpec: f_max, n_mels
    Optional arguments: compute_deltas, compute_delta_deltas
    """

    def __init__(self, feature_type: str = None):
        self.conf = Config('.env.feats')
        self.feature_transformers = {'mfcc': mfcc,
                                     'imfcc': imfcc,
                                     'bfcc': bfcc,
                                     'cqcc': cqcc,
                                     'gfcc': gfcc,
                                     'lfcc': lfcc,
                                     'lpc': lpc,
                                     'lpcc': lpcc,
                                     'msrcc': msrcc,
                                     'ngcc': ngcc,
                                     'pncc': pncc,
                                     'psrcc': psrcc,
                                     'plp': plp,
                                     'rplp': rplp}

        if feature_type is None:
            self.feat_type = self.conf.get_key('feature_type')

    def do_feature_extraction(self, s: torch.Tensor, fs: int):
        """ Feature preparation
        Steps:
        1. Apply feature extraction to waveform
        2. Convert amplitude to dB if required
        3. Append delta and delta-delta features
        """
        if self.feat_type.lower() in self.feature_transformers:
            # Spafe feature selected
            F = self.feature_transformers[self.feat_type](s, fs,
                                                          num_ceps=int(self.config.get('num_ceps')),
                                                          low_freq=int(self.config.get('low_freq')),
                                                          high_freq=int(fs / 2),
                                                          normalize=self.config.get('normalize'),
                                                          pre_emph=self.config.get('pre_emph'),
                                                          pre_emph_coeff=float(self.config.get('pre_emph_coeff')),
                                                          win_len=float(self.config.get('win_len')),
                                                          win_hop=float(self.config.get('win_hop')),
                                                          win_type=self.config.get('win_type'),
                                                          nfilts=int(self.config.get('nfilts')),
                                                          nfft=int(self.config.get('nfft')),
                                                          lifter=float(self.config.get('lifter')),
                                                          use_energy=self.config.get('use_energy') == 'True')
            F = np.nan_to_num(F)
            F = torch.from_numpy(F).T

            if self.conf('compute_deltas') == 'True':
                FD = torchaudio.functional.compute_deltas(F)
                F = torch.cat((F, FD), dim=0)

            if self.conf('compute_delta_deltas') == 'True':
                FDD = torchaudio.functional.compute_deltas(FD)
                F = torch.cat((F, FDD), dim=0)

            return F.T

        else:
            raise ValueError('Feature type not implemented')
