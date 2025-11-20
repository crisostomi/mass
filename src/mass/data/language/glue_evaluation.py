import logging
from pickletools import pylong
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

pylogger = logging.getLogger(__name__)


def remove_special_tokens(tokenizer, token_list: list):
    """
    This function removes special tokens from a list of tokens. It also stops processing
    when it encounters a token with a value of -100.

    Parameters:
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.
        token_list (list): The list of tokens to be processed.

    Returns:
        list: The list of tokens after removing special tokens.
    """
    ret = []
    for token in token_list:
        if token not in tokenizer.all_special_ids and token > 0:
            ret.append(token)
        if token == -100:
            break
    return ret


def evaluate_accuracy(model, batch, tokenizer):

    correct = 0
    total = 0

    with torch.no_grad():
        outputs = model.generate(batch["input_ids"], max_length=10)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels = [
            remove_special_tokens(tokenizer, label_token)
            for label_token in batch["labels"]
        ]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i, j in zip(output_text, labels):
            if i == j:
                correct += 1
            total += 1

    return outputs, correct / total


def evaluate_spearman_rho(model, batch, tokenizer):

    model = model.eval()
    all_preds: List[str] = []
    all_labels: List[str] = []

    with torch.no_grad():
        outputs = model.generate(batch["input_ids"], max_length=10)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels = [
            remove_special_tokens(tokenizer, label_token)
            for label_token in batch["labels"]
        ]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_preds.extend(output_text)
        all_labels.extend(labels)

    from scipy.stats import spearmanr

    def parse_flost(s: str):
        try:
            return float(s)
        except Exception:
            return 0.0

    all_preds = np.array([parse_flost(pred) for pred in all_preds])
    all_labels = np.array([parse_flost(label) for label in all_labels])
    rho = spearmanr(all_preds, all_labels)[0]
    return outputs, rho
