import logging
from pathlib import Path
from pydoc import locate

import torch

from nn_core.serialization import load_model

from mass.modules.encoder import ClassificationHead, ImageEncoder

pylogger = logging.getLogger(__name__)


def load_model_from_artifact(run, artifact_path):
    pylogger.info(f"Loading model from artifact {artifact_path}")

    artifact = run.use_artifact(artifact_path)
    artifact.download()

    ckpt_path = Path(artifact.file())

    model_class = locate(artifact.metadata["model_class"])

    if model_class == ImageEncoder:
        model = model_class(**artifact.metadata)
    elif model_class == ClassificationHead:
        model = model_class(normalize=True, **artifact.metadata)

    model.load_state_dict(torch.load(ckpt_path))

    return model


def load_model_from_disk(model_path, model_name=None) -> ImageEncoder:

    loaded = torch.load(model_path)

    # if it's a statedict, we need to create the model first
    if not isinstance(loaded, ImageEncoder):

        state_dict = loaded

        model = ImageEncoder(model_name)
        model.load_state_dict(state_dict)
        return model

    return loaded


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__
