import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.blank = self.dictionary.bos()

    def decode(self, log_probs, input_lens, labels, label_lens):
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

class BeamSearchCTCDecoder(torch.nn.Module):
    def __init__(self, dictionary, lm_weight=2, word_score=-1):
        super().__init__()
        self.dictionary = dictionary
        self.blank = self.dictionary.bos()
        self.beam_search_decoder = ctc_decoder(
            lexicon='librispeech-lexicon.txt',
            tokens='tokens.txt',
            lm='4-gram-librispeech.bin',
            nbest=1,
            beam_size=100,
            beam_threshold=25,
            beam_size_token=100,
            lm_weight=lm_weight,
            word_score=word_score,
            sil_score=0,
            blank_token='<s>'
        )
    
    def decode(self, emissions, input_lens, labels, label_lens):
        hyp_batch = []
        for emission, in_len in zip(emissions, input_lens):
            emission = emission[:in_len].unsqueeze(0)
            beam_search_result = self.beam_search_decoder(emission)
            hypothesis = " ".join(beam_search_result[0][0].words).strip()
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
