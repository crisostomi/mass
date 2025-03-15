import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os

# Import boilerplate dependencies from your training framework
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

# Import the functions from your modified files.
from tvp.clip_interpret.compute_complete_text_set import run_completeness
from tvp.clip_interpret.compute_text_set_projection import run_text_features

pylogger = logging.getLogger(__name__)

def boilerplate(cfg: DictConfig):
    """
    Full boilerplate initialization mimicking your larger script.
    This sets up tags, restores any checkpoints, and initializes the advanced logger.
    """
    # Enforce and update tags
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))
    num_tasks = len(cfg.eval_datasets)
    cfg.core.tags.append(f"n{num_tasks}")
    cfg.core.tags.append(f"{cfg.nn.module.encoder.model_name}")
    cfg.core.tags.append(f"clip_interpret")
    
    # Initialize template core for resuming experiments
    template_core = NNTemplateCore(restore_cfg=cfg.train.get("restore", None))
    # Initialize the NNLogger with advanced logging configuration
    logger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)
    logger.upload_source()
    return logger, template_core

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="clip_interpret.yaml")
def main(cfg: DictConfig):
    # Initialize boilerplate (advanced logging, tags, etc.)
    logger, template_core = boilerplate(cfg)
    
    # Log the resolved configuration
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)


    name = cfg.text_descriptions.replace('.txt', '')
    output_path = os.path.join(cfg.misc.output_dir, f'{name}_{cfg.model}.npy')

    # Check if output already exists
    if os.path.exists(output_path):
        pylogger.info(f"Output file already exists: {output_path}. Skipping computation.")
    else:
        pylogger.info("Running text feature extraction...")
        run_text_features(cfg)
    
    pylogger.info("Translating into text descriptions...")
    run_completeness(cfg)

    logger.log_configuration(cfg=cfg)
    
    logger.experiment.finish()

if __name__ == "__main__":
    main()