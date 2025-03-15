import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from tvp.clip_interpret.tokenizer.tokenizer import HFTokenizer, tokenize


def get_tokenizer(model_name):
    # TODO: Add support for more tokenizers, e.g. HFtokenizers
    return HFTokenizer(tokenize)