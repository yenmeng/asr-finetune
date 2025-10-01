import tempfile
from pathlib import Path
from typing import List
import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass, is_dataclass
from .hubert_model import (
    HubertConfig,
    HubertModel,
    HubertPretrainingConfig,
)
from .wav2vec2_model import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    AudioPretrainingConfig
)
from .wavlm_model import (
    WavLM, 
    WavLMConfig
)
import torch
import fairseq

logger = logging.getLogger(__name__)

def load_wavlm_ckpt(ckpt_path, no_load=False):
    checkpoint = torch.load(ckpt_path)
    cfg = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg)
    
    if not no_load:
        model.load_state_dict(checkpoint["model"])
        print(f'load pretrained weights from {ckpt_path}')
    else:
        print('training from scratch...')

    return model, cfg

def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str = None, model_type: str = 'hubert', arg_overrides = None):
    from fairseq.data.dictionary import Dictionary

    state, cfg = load_fairseq_ckpt(fairseq_source, arg_overrides)

    if model_type == 'hubert':
        dicts: List[Dictionary] = state["task_state"]["dictionaries"]
        symbols = [dictionary.symbols for dictionary in dicts]
    
    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
    }

    if model_type == 'hubert':
         output_state["dictionaries_symbols"] = symbols

    if output_path is not None:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(output_state, output_path)
    
    return output_state

def load_converted_model(ckpt_path, ckpt_state, model_type: str = 'hubert', no_load: bool = False):
    # ckpt_state = torch.load(ckpt, map_location="cpu")

    required_keys = [
        "task_cfg",
        "model_cfg",
        "model_weight"
    ]
    if model_type == 'hubert':
        required_keys.append("dictionaries_symbols")

    for required_key in required_keys:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt_path} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    if model_type == 'hubert':
        task_cfg = merge_with_parent(HubertPretrainingConfig, ckpt_state["task_cfg"])
        model_cfg = merge_with_parent(HubertConfig, ckpt_state["model_cfg"])
        model = HubertModel(model_cfg, task_cfg, ckpt_state["dictionaries_symbols"])
    elif model_type == 'wav2vec':
        task_cfg = merge_with_parent(AudioPretrainingConfig, ckpt_state["task_cfg"])
        model_cfg = merge_with_parent(Wav2Vec2Config, ckpt_state["model_cfg"])
        model = Wav2Vec2Model(model_cfg)
    
    if not no_load:
        model.load_state_dict(ckpt_state["model_weight"], strict=False)
        print(f'load pretrained weights from {ckpt_path}')
    else:
        print(f'training from scratch...')

    return model, task_cfg

def load_fairseq_ckpt(source: str, override=None):
    from fairseq.checkpoint_utils import load_checkpoint_to_cpu
    from omegaconf import OmegaConf

    source = str(source)
    if source.startswith("http"):
        fairseq_path = _urls_to_filepaths(source)
    else:
        fairseq_path = source

    state = load_checkpoint_to_cpu(fairseq_path, arg_overrides=override)
    cfg = OmegaConf.to_container(state["cfg"])

    assert type(cfg) == dict
    return state, cfg


def merge_with_parent(dc: dataclass, cfg: dict):
    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)
    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys: 
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)
