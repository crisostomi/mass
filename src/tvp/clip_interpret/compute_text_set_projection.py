import time
import numpy as np
import torch
import os
import json
import glob
import datetime
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import open_clip
import logging
from tvp.clip_interpret.tokenizer.utils import get_tokenizer

pylogger = logging.getLogger(__name__)


def get_text_features(model, tokenizer, lines, device, batch_size, amp=True):
    """
    Returns zero-shot text embeddings for each class.
    """
    autocast = torch.cuda.amp.autocast
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for i in tqdm.trange(0, len(lines), batch_size):
            texts = lines[i:i+batch_size]
            texts = tokenizer(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            zeroshot_weights.append(class_embeddings.detach().cpu())
        zeroshot_weights = torch.concatenate(zeroshot_weights, dim=0)
    return zeroshot_weights

def run_text_features(cfg):
    """
    Run the text feature extraction routine using the Hydra config.
    Expects the following config attributes:
      - batch_size, model, pretrained, data_path, num_workers, output_dir, device
    """
    model, _, preprocess = open_clip.create_model_and_transforms(cfg.model, pretrained=cfg.pretrained)
    tokenizer = get_tokenizer(cfg.model)
    model.to(cfg.device)
    model.eval()
    vocab_size = model.vocab_size

    pylogger.info(f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    pylogger.info(f"Vocab size: {vocab_size}")
    
    with open(cfg.data_path, 'r') as f:
        lines = f.readlines()

    # Remove newline characters and extra spaces
    lines = [line.strip() for line in lines if line.strip()]

    # Get text features
    features = get_text_features(model, tokenizer, lines, cfg.device, cfg.batch_size)
    base, name = os.path.split(cfg.misc.description_dir)
    name = name.replace('.txt', '')
    output_path = os.path.join(cfg.misc.output_dir, f'{name}_{cfg.model}.npy')
    with open(output_path, 'wb') as f:
        np.save(f, features.numpy())