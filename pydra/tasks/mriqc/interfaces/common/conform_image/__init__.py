import attrs
from fileformats.generic import Directory, File
import logging
import numpy as np
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, MultiInputObj, SpecInfo
from pydra.engine.task import FunctionTask
import pydra.mark
from .conform_image import ConformImage
import typing as ty


logger = logging.getLogger(__name__)


NUMPY_DTYPE = {
    1: np.uint8,
    2: np.uint8,
    4: np.uint16,
    8: np.uint32,
    64: np.float32,
    256: np.uint8,
    1024: np.uint32,
    1280: np.uint32,
    1536: np.float32,
}

OUT_FILE = "{prefix}_conformed{ext}"
