import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import fairseq
from .utils import load_converted_model, load_and_convert_fairseq_ckpt, load_wavlm_ckpt

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)

class Wrapper(nn.Module):
    def __init__(self, ckpt, train_config, output_dim):
        super().__init__()

        if 'hubert' in ckpt.lower():
            self.model_type = 'hubert' 
        elif 'wav2vec' in ckpt.lower():
            self.model_type = 'wav2vec'
        elif 'wavlm' in ckpt.lower():
            self.model_type = 'wavlm'
        else:
            raise NotImplementedError
        print(f"model type: {self.model_type}")
        
        no_load = train_config.get('no_load', False)

        if self.model_type == 'wavlm':
            ckpt_state = torch.load(ckpt, map_location='cpu')
            model, task_cfg = load_wavlm_ckpt(ckpt, no_load=no_load)
        else:
            arg_overrides = {
                "activation_dropout": train_config["activation_dropout"],
                "feature_grad_mult" : train_config['feature_grad_mult'],
                "encoder_layerdrop": train_config["layerdrop"],
                "mask_prob": train_config["mask_prob"],
                "mask_channel_length": train_config["mask_channel_length"],
                "mask_channel_prob": train_config["mask_channel_prob"],
            }
            ckpt_state = load_and_convert_fairseq_ckpt(ckpt,  model_type=self.model_type, arg_overrides=arg_overrides)
            model, task_cfg = load_converted_model(ckpt, ckpt_state, model_type=self.model_type, no_load=no_load)
        self.config = train_config
        self.encoder = model
        self.task_cfg = task_cfg
        self.model_cfg = ckpt_state['model_cfg'] if ckpt_state.get('model_cfg') else ckpt_state['cfg']

        # self.encoder.feature_grad_mult = train_config['feature_grad_mult']
        # self.encoder.mask_prob = train_config['mask_prob']
        # self.encoder.mask_channel_prob = train_config['mask_channel_prob']
        # self.encoder.mask_channel_length = train_config['mask_channel_length']
        # self.encoder.encoder.layerdrop = train_config['layerdrop']
        self.freeze_finetune_updates = train_config['freeze_finetune_updates']
        self.freeze_layers = train_config.get('freeze_layers', 0)

        if self.freeze_layers > 0:
            for i in range(self.freeze_layers):
                model.encoder.layers[i].requires_grad_(False)
        
        self.proj = nn.Linear(self.model_cfg['encoder_embed_dim'], output_dim, bias=True)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, mask=False, step=None, tgt_layer=None):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wavs = pad_sequence(wavs, batch_first=True)
        
        if self.training:
            assert step is not None
            if step <= self.freeze_finetune_updates:
                with torch.no_grad():
                    features, feat_padding_mask, feat_lengths = self.encoder.extract_features(
                        padded_wavs,
                        source_lengths=wav_lengths,
                        padding_mask=wav_padding_mask,
                        mask=mask,
                        output_layer=tgt_layer
                    )
            else:
                features, feat_padding_mask, feat_lengths = self.encoder.extract_features(
                        padded_wavs,
                        source_lengths=wav_lengths,
                        padding_mask=wav_padding_mask,
                        mask=mask,
                        output_layer=tgt_layer
                    )
            x = self.proj(features)

        else:
            with torch.no_grad():
                features, feat_padding_mask, feat_lengths = self.encoder.extract_features(
                    padded_wavs,
                    source_lengths=wav_lengths,
                    padding_mask=wav_padding_mask,
                    mask=mask,
                    output_layer=tgt_layer
                )
        
                x = self.proj(features)

        return x, feat_lengths
