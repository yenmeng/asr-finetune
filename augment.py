from typing import Dict, List, Optional, Callable, Union, Tuple, Set
import os
import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from itertools import groupby
import torch
import torchaudio

class Augment:
    def __init__(self):
        self.sample_rate = 16000
        self.noise_files = glob(os.path.join('/disk/scratch/s2522924/musan/noise', '**/*.wav'), recursive=True)
        print(f"Found {len(self.noise_files)} noise clips")

    def add_gaussian_noise(self, signal, snr):
        noise = torch.randn(signal.shape[0])
        coeff = self._snr(signal, noise, snr)
        noise *= coeff
        signal += noise
        return signal

    def add_real_noise(self, signal, noise, snr):
        signal_len = signal.shape[0]
        noise_len = noise.shape[0]
        if signal_len <= noise_len:
            start = random.randint(0, noise_len - signal_len)
            noise = noise[start: start+signal_len]
        else:
            n_repeat = signal_len // noise_len + 1
            noise = np.repeat(noise, n_repeat)
            noise = noise[: signal_len]
        coeff = self._snr(signal, noise, snr)
        noise *= coeff

        signal += noise
        return signal

    def apply_aug(self, signal, aug_type, target_sample_rate=None, snr=None):
        if target_sample_rate is not None and target_sample_rate != self.sample_rate:
            self._resample(signal, target_sample_rate)

        if aug_type == 'g':
            assert snr is not None
            return self.add_gaussian_noise(signal, snr)

        elif aug_type == 'm':
            noise_path = random.choice(self.noise_files)
            noise, sr = torchaudio.load(noise_path)
            noise = noise.squeeze(0)
            return self.add_real_noise(signal, noise, snr)

    def _snr(self, signal, noise, snr):
        signal_pow = torch.sum(torch.pow(signal, 2))
        noise_pow = torch.sum(torch.pow(noise, 2))
        return ((signal_pow / noise_pow) * 10 ** (-snr / 10)) ** 0.5

    def _resample(self, signal, target_sample_rate):
        resampler = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)
        return resampler(signal)
