import os
import sys
import numpy as np
import math
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch.nn.utils.rnn import pad_sequence

class WavDataset(torch.utils.data.Dataset):
    
    def __init__(self, split, dictionary, device='cuda:0'):
        super(WavDataset, self).__init__()
        self.device = device
        
        root = '/disk/scratch/s2522924'
        path = 'LibriSpeech'
        self.file_list = []
        self.trans_list = []
        for s in split:
            self.file_list += sorted(glob(os.path.join(root, path, s, "**/*.flac"), recursive=True))
            self.trans_list += sorted(glob(os.path.join(root, path, s, "**/*.trans.txt"), recursive=True))
        self.label_dict = {}
        for trans in self.trans_list:
            lines = open(trans, 'r').readlines()
            for l in lines:
                fid, text = l.split(' ', 1)
                self.label_dict[fid] = text
        
        self.dictionary = dictionary
        # temp hard code
        # filtered = ['/disk/scratch/s2522924/LibriSpeech/dev-clean/2078/142845/2078-142845-0005.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/2902/9006/2902-9006-0005.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/2902/9006/2902-9006-0007.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/2902/9006/2902-9006-0015.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/2902/9006/2902-9006-0018.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/422/122949/422-122949-0010.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/422/122949/422-122949-0013.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/5338/24615/5338-24615-0002.flac',
        # '/disk/scratch/s2522924/LibriSpeech/dev-clean/8842/304647/8842-304647-0002.flac']
        # for s in split:
        #     if s == 'dev-clean':
        #         for f in filtered:
        #             self.file_list.remove(f)
        
    def _load_audio(self, path):
        audio, sample_rate = torchaudio.load(path)
        assert sample_rate == 16000
        audio = audio.squeeze()
        return audio

    def _load_transcript(self, fid):
        transcript = self.label_dict[fid]
        transcript = transcript.upper().strip().replace(" ", "|")
        transcript =  " ".join(list(transcript)) + " |"
        text_tokens = self.dictionary.encode_line(transcript, line_tokenizer=lambda x: x.split()).long() 
        return text_tokens

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fpath = self.file_list[index]
        fid = fpath.split('/')[-1].split('.')[0]
        wav = self._load_audio(fpath)
        text = self._load_transcript(fid)
        return wav, text, fid

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        feats, labels, fileids = batch
        feat_lengths = [len(f) for f in feats]
        label_lengths = [len(f) for f in labels]
        return feats, feat_lengths, labels, label_lengths, fileids
