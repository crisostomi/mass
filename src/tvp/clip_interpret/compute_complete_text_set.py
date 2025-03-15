import time
import numpy as np
import torch
import os
import einops
import tqdm
import open_clip
import wandb
import logging
from pathlib import Path

from tvp.utils.io_utils import load_model_from_disk
from tvp.utils.utils import compute_task_dict
from tvp.task_vectors.task_singular_vectors import get_svd_dict

pylogger = logging.getLogger(__name__)

@torch.no_grad()
def replace_with_iterative_removal(data, text_features, texts, iters, device):
    results = []
    vh = data # in our case we already have "vectors"...
    text_features = (
        vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(text_features.T).T
    )  # Project the text to the span of W_OV
    data = torch.from_numpy(data).float().to(device)
    mean_data = data.mean(dim=0, keepdim=True)
    data = data - mean_data
    reconstruct = einops.repeat(mean_data, "A B -> (C A) B", C=data.shape[0])
    reconstruct = reconstruct.detach().cpu().numpy()
    text_features = torch.from_numpy(text_features).float().to(device)
    for i in range(iters):
        projection = data @ text_features.T
        projection_std = projection.std(axis=0).detach().cpu().numpy()
        top_n = np.argmax(projection_std)
        results.append(texts[top_n])
        text_norm = text_features[top_n] @ text_features[top_n].T
        reconstruct += (
            (
                (data @ text_features[top_n] / text_norm)[:, np.newaxis]
                * text_features[top_n][np.newaxis, :]
            )
            .detach()
            .cpu()
            .numpy()
        )
        data = data - (
            (data @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
        text_features = (
            text_features
            - (text_features @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
    return reconstruct, results

@torch.no_grad()
def run_completeness(cfg):
    """
    Run the SVD-based completeness procedure using the Hydra config.
    Expects the following config attributes:
      - model, input_dir, output_dir, text_descriptions, text_dir,
        dataset, num_of_last_layers, w_ov_rank, texts_per_head, device
    """
    
    zeroshot_encoder_statedict = load_model_from_disk(cfg.misc.pretrained_checkpoint)
    
    finetuned_name = (
        lambda name: Path(cfg.misc.ckpt_path) / f"{name}Val" / "nonlinear_finetuned.pt"
    )
    finetuned_models = {
        dataset: load_model_from_disk(finetuned_name(dataset))
        for dataset in cfg.task_vectors.to_apply
    }
    
    task_dicts = {}
    for dataset in cfg.task_vectors.to_apply:
        task_dicts[dataset] = compute_task_dict(
            zeroshot_encoder_statedict, finetuned_models[dataset]
        )
        del finetuned_models[dataset]  # Delete one model at a time
        torch.cuda.empty_cache()

    svd_dict = get_svd_dict(
        task_dicts, cfg.eval_datasets, cfg.misc.svd_path, cfg.svd_compress_factor
    )

    model, _, preprocess = open_clip.create_model_and_transforms(cfg.model, pretrained="openai", cache_dir=cfg.misc.openclip_cachedir)
    model.to(cfg.device)
    all_images = set()

    # Load text features from file
    name = cfg.text_descriptions.replace('.txt', '')
    text_features_path = os.path.join(cfg.misc.output_dir, f'{name}_{cfg.model}.npy')

    with open(text_features_path, "rb") as f:
        text_features = np.load(f)
    pylogger.info(f"Loaded text features from {text_features_path}")

    # Load text descriptions (each line is one text)
    text_file = os.path.join(cfg.misc.description_dir, f"{cfg.text_descriptions}")
    with open(text_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    pylogger.info(f"Loaded text descriptions from {text_file}")

    # Prepare a wandb Table for fancy terminal logging
    results_table = wandb.Table(columns=["Task", "Top Texts"])

    # Optionally also write the results to a file
    output_file = os.path.join(
        cfg.misc.output_dir,
        f"{cfg.dataset}_completeness_{cfg.text_descriptions}_top_{cfg.texts_per_task}_heads_{cfg.model}.txt",
    )
    out_f = open(output_file, "w")
    
    for task in svd_dict.keys():
        pylogger.info(f"Processing Task: {task}")
        out_f.write(f"------------------\n")
        out_f.write(f"V for task {task}\n")
        out_f.write(f"------------------\n")
        
        # Retrieve SVD components for the current task
        u = svd_dict[task][cfg.layer]['u'].to(cfg.device)
        s = torch.diag_embed(svd_dict[task][cfg.layer]['s']).to(cfg.device)
        v = svd_dict[task][cfg.layer]['v'].to(cfg.device)
        # pylogger.info(f"v shape for task {task}: {v.shape}")

        # Compute the projected matrix
        v_proj = s @ v @ model.visual.proj

        # Apply the iterative removal procedure
        reconstruct, images = replace_with_iterative_removal(
            v_proj.cpu().numpy(),
            text_features,
            lines,
            cfg.texts_per_task,
            cfg.device,
        )

        all_images |= set(images)
        for text in images:
            out_f.write(f"{text}\n")
        results_table.add_data(task, "\n".join(images))
        pylogger.info(f"Task {task}: {images}")

    out_f.close()
    wandb.log({"interpret_results": results_table})
