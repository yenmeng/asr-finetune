import os
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
from dictionary import Dictionary
from dataset import WavDataset
import jiwer
from torch.nn.utils.rnn import pad_sequence
from lr_schedule.cosine_lr_schedule import get_cosine_schedule_with_warmup
from lr_schedule.tri_stage_lr_schedule import TriStageLRScheduler
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Runner():
    def __init__(self, config, paras, device):
        # general settings
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.hparas_config = config['hparas']
        self.paras = paras
        self.device = device
        
        # logger
        self.logdir = os.path.join(paras.log_dir, paras.run_name)
        self.writer = SummaryWriter(self.logdir)

        # model
        self.model_name = self.model_config.get('name', 'ssl')
        self.pretrained_path = self.paras.ckpt_path
        if self.paras.resume_ckpt is not None:
            self.init_ckpt = torch.load(self.paras.resume_ckpt, map_location=self.device)
            self.pretrained_path = self.init_ckpt['pretrained_path']
        else:
            self.init_ckpt = None
        self.tgt_layer = self.model_config.get('tgt_layer', None)

        # hyperparameters
        self.fp16 = self.hparas_config.get('fp16', False)           
        self.grad_clip = self.hparas_config.get('grad_clip', 5)
        self.lr_schedule = self.hparas_config.get('lr_schedule', False)
        self.batch_size = self.hparas_config.get('batch_size', 16)
        self.num_workers = self.hparas_config.get('num_workers', 8)
        self.gradient_accumulate_steps = self.hparas_config.get('gradient_accumulate_steps', 1)
        self.step = 0
        self.cur_epoch = 0
        self.epoch_loss = 100.0
        self.save_step = self.hparas_config.get('save_step', -1)
        self.plot_step = self.hparas_config.get('plot_step', 1000)
        self.eval_step = self.hparas_config.get('eval_step', 1000)
        self.save_every_epoch = self.hparas_config.get('save_epoch', 1)
        
        # aug
        self.apply_mask = self.model_config.get('apply_mask', False)
        self.data_aug = self.data_config.get('data_aug', False)

        # set dictionary
        self.dictionary = Dictionary.load('./dict.ltr.txt')
        self.blank = self.dictionary.bos()

        # load data
        logger.info('loading data...')
        self.train_loader, self.dev_loader = self.load_data()
        
        self.best_score = float('inf')
        
    def verbose(self, msg):
        print(f"[INFO] - {msg}")

    def load_weight(self, model, name):
        assert self.init_ckpt is not None
        init_weight = self.init_ckpt.get(name)
        model.load_state_dict(init_weight)
        logger.info(f'Resume training: loading {name} weights from {self.paras.resume_ckpt}')
        return model
    
    def init_weight(self):
        self.optimizer = self.load_weight(self.optimizer, 'optimizer')
        self.model = self.load_weight(self.model, 'model')

    def save_checkpoint(self, mode='step', name=None):
        if name is None:
            if mode == 'step':
                name = f"step-{self.step}.ckpt"
            elif mode == 'epoch':
                name = f"epoch-{self.cur_epoch}.ckpt"
        ckpt_path = os.path.join(self.logdir, name)
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "task_cfg": self.model.task_cfg,
            # "model_cfg": self.model.model_cfg,
            "config": self.config,
            "pretrained_path": self.pretrained_path,
            "step": self.step,
            "epoch": self.cur_epoch
        }
        torch.save(ckpt, ckpt_path)
        logger.info(f'saving checkpoint @ step {self.step}(epoch {self.cur_epoch})')

    def set_model(self):
        if self.model_name == 'ssl':
            from model.ssl.wrapper import Wrapper
        elif self.model_name == 'whisper':
            from model.whisper.wrapper import Wrapper
        else:
            raise NotImplementedError

        self.model = Wrapper(ckpt=self.pretrained_path, train_config=self.model_config, output_dim=len(self.dictionary.symbols))
        self.model = self.model.to(self.device) 
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"total parameters: {total_params}")
        logger.info(f"model:\n{self.model}")

    def training_setup(self):
        # set training steps
        n_epochs = self.hparas_config['epoch']
        if n_epochs > 0: 
            self.total_steps = int(n_epochs * len(self.train_loader) / self.gradient_accumulate_steps)
            logger.info(f'Training for {n_epochs} epochs, which is equivalent to {self.total_steps} steps')
        else:
            self.total_steps = self.hparas_config['total_steps']
            n_epochs = int(self.total_steps * self.gradient_accumulate_steps / len(self.train_loader))
            logger.info(f'Training for {self.total_steps} steps, which is approximately {n_epochs} epochs')
        self.steps_per_epoch = len(self.train_loader)// self.gradient_accumulate_steps
        logger.info(f'Steps per epoch: {self.steps_per_epoch} steps')

        self.criterion = nn.CTCLoss(
            blank=self.blank, zero_infinity=True
        )
        # set optimizer
        if self.hparas_config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.hparas_config['lr'], 
                                             momentum=self.hparas_config['momentum'],
                                             weight_decay=self.hparas_config['weight_decay'])
        else:
            self.optimizer = eval(f"torch.optim.{self.hparas_config['optimizer']}")(self.model.parameters(), lr=self.hparas_config['lr'])
        
        if self.init_ckpt is not None:
            self.init_weight()
        else:
            self.save_checkpoint(mode='step') # init

        # set scheduler
        if self.lr_schedule:
            # self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, round(self.total_steps * self.hparas_config['warmup_proportion']), self.total_steps)
             self.scheduler = TriStageLRScheduler(
                self.optimizer,  
                peak_lr=self.hparas_config['lr'], 
                init_lr_scale=0.01, 
                final_lr_scale=0.05,
                phase_ratio=[0.1, 0.4, 0.5],
                total_steps=self.total_steps,
            )
        else:
            self.scheduler = None

    def load_data(self):
        train_loader = self._get_train_loader()
        dev_loader = self._get_dev_loader()
        return train_loader, dev_loader

    def _get_train_loader(self):
        train_set = WavDataset(split=self.data_config['train_split'], dictionary=self.dictionary, device=self.device, augment=self.data_aug)
        train_loader = DataLoader(train_set, 
                                  batch_size=self.batch_size, 
                                  collate_fn=train_set.collate_fn, 
                                  pin_memory=True, 
                                  num_workers=self.num_workers, 
                                  shuffle=True)
        return train_loader

    def _get_dev_loader(self):
        dev_set = WavDataset(split=self.data_config['dev_split'],dictionary=self.dictionary, device=self.device, augment=False)
        dev_loader = DataLoader(dev_set, 
                                  batch_size=1,
                                  collate_fn=dev_set.collate_fn, 
                                  pin_memory=True, 
                                  num_workers=self.num_workers, 
                                  shuffle=False)
        return dev_loader

    def get_log_probs(self, feats, mask=False):
        logits, feat_lengths = self.model(feats, mask=mask, step=self.step, tgt_layer=self.tgt_layer)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        return log_probs, feat_lengths

    def _decode(self, log_probs, input_lens, labels, label_lens):
        """Decoder that take log probabilities as input and outputs decoded seq"""
        hyp_batch = []
        for log_prob, in_len in zip(log_probs, input_lens):
            log_prob = log_prob[:in_len].unsqueeze(0)
            pred_token_ids = log_prob.argmax(dim=-1).unique_consecutive()
            pred_token_ids = pred_token_ids[pred_token_ids != self.blank].tolist()
            hypothesis = self.dictionary.string(pred_token_ids)
            hypothesis = hypothesis.replace(" ", "").replace("|", " ").strip()
            hyp_batch.append(hypothesis)
        
        ref_batch = []
        for label in labels:
            label_idx = (label != self.dictionary.pad()) & (
                label != self.dictionary.eos()
            )
            label_tokens = label[label_idx].tolist()
            reference = self.dictionary.string(label_tokens)
            reference = reference.replace(" ", "").replace("|", " ").strip()
            ref_batch.append(reference)
            
        return hyp_batch, ref_batch
    
    def exec(self):
        self.set_model()
        self.training_setup()
                
        if self.fp16:
            logger.info('enable fp16 training')
            scaler = torch.amp.GradScaler('cuda')

        pbar = tqdm(total=self.total_steps, dynamic_ncols=True, desc='overall')
        pbar.n = 0
        if self.init_ckpt is not None:
            pbar.n = self.init_ckpt.get('step')
            self.cur_epoch = self.init_ckpt.get('epoch')
            self.step = self.init_ckpt.get('step')

        train_loss = 0
        epoch_loss = []
        backward_step = 0

        while pbar.n < pbar.total:
            # train
            for batch in tqdm(self.train_loader, dynamic_ncols=True, desc='train'):
                
                try:
                    self.model.train()
                    if pbar.n >= pbar.total:
                        break
                    self.step = pbar.n + 1
 
                    feats, feat_lengths, labels, label_lengths, fnames = batch
                    feats = [feat.to(self.device) for feat in feats]
                    labels_len = torch.LongTensor([len(label) for label in labels]).to(self.device)
                    labels = pad_sequence(
                        labels,
                        batch_first=True,
                        padding_value=self.dictionary.pad(),
                    )
                    labels = labels.to(self.device)
                    
                    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
                        log_probs, log_probs_len = self.get_log_probs(feats, mask=self.apply_mask)
                        loss = self.criterion(
                            log_probs.transpose(0, 1),  # (N, T, C) -> (T, N, C)
                            labels,
                            log_probs_len,
                            labels_len,
                        )
                        
                    if self.gradient_accumulate_steps > 1:
                        loss = loss / self.gradient_accumulate_steps

                    if self.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {self.step}')
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise
                
                train_loss += loss.item()

                # whether to accumulate gradient
                backward_step += 1
                if backward_step % self.gradient_accumulate_steps > 0:
                    continue

                if self.fp16:
                    scaler.unscale_(self.optimizer)
                
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                if self.fp16:
                    scaler.step(self.optimizer)
                    prev_scaler = scaler.get_scale()
                    scaler.update()
                    new_scaler = scaler.get_scale()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.lr_schedule:
                    if self.fp16:
                        if new_scaler >= prev_scaler:
                            self.scheduler.step()
                    else:
                        self.scheduler.step()
            
                if self.step % self.plot_step == 0 or pbar.n == pbar.total -1:
                    if self.step % self.plot_step == 0:
                        train_loss /= self.plot_step
                    else:
                        train_loss /= (self.step % self.plot_step)
        
                    self.writer.add_scalar('Loss', train_loss, self.step)
                    # print(train_loss, self.step)
                    epoch_loss.append(train_loss)
                    train_loss = 0
                    self.writer.add_scalar('Gradient Norm', grad_norm, self.step)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.step)
                
                if self.step % self.eval_step == 0:
                    wer, cer = self.evaluate()
                    # self.writer.add_scalar('dev WER', wer, self.step)
                    # self.writer.add_scalar('cer CER', cer, self.step)
                    if wer < self.best_score:
                        self.best_score = wer
                        self.save_checkpoint(mode='step', name='dev-best.ckpt')

                # unfreeze encoder
                # if self.step >= self.freeze_finetune_updates:
                #     self.model.encoder.requires_grad_(True)

                if (self.step % self.steps_per_epoch == 0) and (backward_step % self.gradient_accumulate_steps == 0):
                    self.cur_epoch = self.step // self.steps_per_epoch
                    
                    if self.cur_epoch > 0:
                        epoch_loss = sum(epoch_loss) / len(epoch_loss)
                        logger.info(f"avg loss @ epoch {self.cur_epoch}: {epoch_loss}")
                        epoch_loss = []

                    if self.cur_epoch > 0 and self.cur_epoch % self.save_every_epoch == 0:
                        self.save_checkpoint(mode='epoch')


                pbar.update(1)
        
        self.save_checkpoint(mode='step')
        pbar.close()


    def evaluate(self):
        self.model.eval()
        hypotheses = []
        references = []
        files = []
        for batch in tqdm(self.dev_loader, dynamic_ncols=True, desc='dev'):
            feats, feat_lengths, labels, label_lengths, fnames = batch
            feats = [feat.to(self.device) for feat in feats]

            with torch.no_grad():
                log_probs, log_prob_lens = self.get_log_probs(feats, mask=False)
            
            hyp_batch, ref_batch = self._decode(log_probs.float().contiguous().cpu(), log_prob_lens, labels, label_lengths)
            hypotheses.extend(hyp_batch)
            references.extend(ref_batch)
            files.extend(fnames)

        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        print(f"WER: {wer * 100:.2f}")
        print(f"CER: {cer * 100:.2f}")
        return wer * 100, cer * 100






                
                
