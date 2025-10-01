# CTC Finetuning
Simple implementation for finetuning a pretrained SSL model with CTC (does not require using the fairseq pipeline for training and inference).

support models: wav2vec2.0, HuBERT, WavLM


## Dependencies
```
pip install fairseq
pip install editdistance
```
install [kenlm](https://github.com/kpu/kenlm) and [flashlight](https://github.com/flashlight/text) for beam search decoding
