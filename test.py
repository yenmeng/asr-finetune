import os
import sys
sys.path.append('/home/s2522924/asr-finetune/model/ssl')
import numpy as np
import math
import yaml
from tqdm import tqdm
from functools import partial
from shutil import copyfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.ssl.wrapper import Wrapper
from dictionary import Dictionary
from dataset import WavDataset
from decoder import GreedyCTCDecoder, BeamSearchCTCDecoder
import jiwer
import editdistance

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {DEVICE}')

def WER(hyp, ref):
    total_length = 0
    error = 0
    for h, r in zip(hyp, ref):
        h = h.split(' ')
        r = r.split(' ')
        error += float(editdistance.eval(h, r))
        total_length += len(r)
    return error / total_length

def CER(hyp, ref):
    total_length = 0
    error = 0
    for h, r in zip(hyp, ref):
        error += float(editdistance.eval(h, r))
        total_length += len(r)
    return error / total_length

dictionary = Dictionary.load('./dict.ltr.txt')
dev_set = WavDataset(split=['dev-clean'], dictionary=dictionary, device=DEVICE)
dev_loader = DataLoader(dev_set, 
                            batch_size=1, 
                            collate_fn=dev_set.collate_fn, 
                            pin_memory=True, 
                            num_workers=8, 
                            shuffle=False)

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
print(ckpt.keys())
model = Wrapper(ckpt=ckpt["pretrained_path"], train_config=ckpt['config']['model'], output_dim=len(dictionary.symbols))
model.load_state_dict(ckpt['model'])
model = model.to(DEVICE)
model.requires_grad_(False)
model.eval()
print(model)
model_config = ckpt['config']['model']
tgt_layer = model_config.get('tgt_layer', None)
print(f'final layer: {tgt_layer}')

decoder = GreedyCTCDecoder(dictionary) 
# decoder = BeamSearchCTCDecoder(dictionary)

hypotheses = []
references = []
files = []
for i, batch in enumerate(tqdm(dev_loader, dynamic_ncols=True, desc='test')):
    feats, feat_lengths, labels, label_lengths, fnames = batch
    feats = [feat.to(DEVICE) for feat in feats]
    
    with torch.no_grad():
        logits, feat_lengths = model(feats, tgt_layer=tgt_layer)
        # log_probs = nn.functional.log_softmax(logits, dim=-1)
        # hyp_batch, ref_batch = decoder.decode(log_probs.float().contiguous().cpu(), feat_lengths, labels, label_lengths)
        hyp_batch, ref_batch = decoder.decode(logits.float().contiguous().cpu(), feat_lengths, labels, label_lengths)

    hypotheses.extend(hyp_batch)
    references.extend(ref_batch)
    files.extend(fnames)

for i, (name, hyp, ref) in enumerate(zip(files, hypotheses, references)):
    print(name)
    print(f"hyp: {hyp}")
    print(f"ref: {ref}\n")
    if i == 5:
        break

# wer = jiwer.wer(references, hypotheses)
# cer = jiwer.cer(references, hypotheses)
wer = WER(references, hypotheses)
cer = CER(references, hypotheses)
print(f"WER: {wer * 100:.2f}")
print(f"CER: {cer * 100:.2f}")


        
    
