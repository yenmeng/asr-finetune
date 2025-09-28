import logging
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .model import ModelDimensions, AudioEncoder, sinusoids
from .audio import log_mel_spectrogram, pad_or_trim

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)

class Wrapper(nn.Module):
    def __init__(self, ckpt, train_config, output_dim, **kwargs):
        super().__init__(**kwargs)

        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        dims = ModelDimensions(**checkpoint["dims"])
        self.task_cfg = dims
        self.model_cfg = dims
        self.model = AudioEncoder(
            self.task_cfg.n_mels,
            self.task_cfg.n_audio_ctx,
            self.task_cfg.n_audio_state,
            self.task_cfg.n_audio_head,
            self.task_cfg.n_audio_layer,
        )
        new_dict = {k.replace('encoder.', ''):v for (k, v) in checkpoint["model_state_dict"].items() if 'encoder' in k}
        # temp hard code
        new_dict.pop('positional_embedding')
        self.model.load_state_dict(new_dict, strict=False)
        print(f'load checkpoint from path: {ckpt}')
        
        self.freeze_finetune_updates = train_config['freeze_finetune_updates']
        self.proj = nn.Linear(self.model_cfg.n_audio_state, output_dim, bias=True)


    def get_downsample_rates(self, key: str) -> int:
        return 320

    def _get_conv_output_length(self, input_length, kernel_size=3, stride=2, padding=1):
        return int(np.floor((input_length - kernel_size) / stride + 1))

    def forward(self, wavs, mask=False, step=None):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        mels = []
        feat_lengths = []
        for wav in wavs:
            # wav = pad_or_trim(wav.flatten())
            mel = log_mel_spectrogram(wav.flatten(), self.task_cfg.n_mels)
            output_length = self._get_conv_output_length(mel.size(-1))
            feat_lengths.append(output_length)
            mels.append(mel.transpose(0, 1))
        mels = pad_sequence(mels, batch_first=True)
        mels.to(device)
        mels = mels.permute(0, 2, 1)
        feat_lengths = torch.Tensor(feat_lengths).long().to(device)

        if self.training:
            assert step is not None
            if step <= self.freeze_finetune_updates or self.freeze_finetune_updates < 0:
                with torch.no_grad():
                    features = self.model(mels)
            else:
                features = self.model(mels)
            x = self.proj(features)

        else:
            with torch.no_grad():
                features = self.model(mels)
                x = self.proj(features)

        return x, feat_lengths

