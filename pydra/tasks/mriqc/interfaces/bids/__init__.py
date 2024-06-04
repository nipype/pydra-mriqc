import attrs
from fileformats.generic import Directory, File
import logging
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, MultiInputObj, SpecInfo
from pydra.engine.task import FunctionTask
import pydra.mark
from .iqm_file_sink import IQMFileSink
import typing as ty


logger = logging.getLogger(__name__)


def _process_name(name, val):

    if "." in name:
        newkeys = name.split(".")
        name = newkeys.pop(0)
        nested_dict = {newkeys.pop(): val}
        for nk in reversed(newkeys):
            nested_dict = {nk: nested_dict}
        val = nested_dict
    return name, val
