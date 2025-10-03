import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse
import shutil
import torch
from runner import Runner
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--run_name', type=str, default='exp')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('-p', '--ckpt_path', type=str, default='/home/s2522924/checkpoints/fairseq/wav2vec_small.pt')
    parser.add_argument('-c', '--config', type=str, default='config/apc_config.yaml', help='training config')
    parser.add_argument('-r', '--resume_ckpt', type=str, default=None)
    #seed
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f'DEVICE: {DEVICE}')

    paras = get_args()
    same_seeds(paras.seed)
    config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)
    os.makedirs(os.path.join(paras.log_dir, paras.run_name), exist_ok=True)

    try:
        shutil.copyfile(paras.config, os.path.join(paras.log_dir, paras.run_name, 'config.yaml'))
    except shutil.SameFileError:
        pass

    runner = Runner(config, paras, device=DEVICE)
    runner.exec() 
