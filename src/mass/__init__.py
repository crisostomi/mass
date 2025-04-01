import logging
import os

from nn_core.console_logging import NNRichHandler
from nn_core.common import PROJECT_ROOT

# Required workaround because PyTorch Lightning configures the logging on import,
# thus the logging configuration defined in the __init__.py must be called before
# the lightning import otherwise it has no effect.
# See https://github.com/PyTorchLightning/pytorch-lightning/issues/1503
lightning_logger = logging.getLogger("lightning.pytorch")
# Remove all handlers associated with the lightning logger.
for handler in lightning_logger.handlers[:]:
    lightning_logger.removeHandler(handler)
lightning_logger.propagate = True

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        NNRichHandler(
            rich_tracebacks=True,
            show_level=True,
            show_path=True,
            show_time=True,
            omit_repeated_times=True,
        )
    ],
)

try:
    from ._version import __version__ as __version__
except ImportError:
    import sys

    print(
        "Project not installed in the current env, activate the correct env or install it with:\n\tpip install -e .",
        file=sys.stderr,
    )
    __version__ = "unknown"


# hack to reuse models finetuned with code having relative paths instead of properly installed modules

import sys
import mass
import mass.data
import mass.modules
import mass.modules.heads
import types

src = types.ModuleType("src")

# Insert the alias into sys.modules so that any import of "src" resolves to this module.
sys.modules["src"] = src
sys.modules["src.data"] = mass.data
sys.modules["src.modules"] = mass.modules
sys.modules["src.models.modeling"] = mass.modules.heads


from dotenv import dotenv_values, load_dotenv

load_dotenv()


os.environ["WANDB_DIR"] = str(PROJECT_ROOT / "wandb")
