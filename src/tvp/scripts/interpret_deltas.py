import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb

# Import boilerplate dependencies from your training framework
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.callbacks import NNTemplateCore
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

@hydra.main(config_path="conf", config_name="clip_interpret.yaml")
def main(cfg: DictConfig):
    # Initialize boilerplate (advanced logging, tags, etc.)
    logger, template_core = boilerplate(cfg)
    
    # Log the resolved configuration
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    pylogger.info("Configuration: %s", resolved_cfg)
    
    # Instead of building an argparse.Namespace, simply pass the configuration to your routines.
    if cfg.compute_text_features:
        pylogger.info("Running text feature extraction...")
        run_text_features(cfg)
    
    pylogger.info("Running completeness SVD removal...")
    run_completeness(cfg)
    pylogger.info("Completed all routines.")
    
    logger.experiment.finish()

if __name__ == "__main__":
    main()