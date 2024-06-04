import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
import pydra.mark
import scipy.ndimage as nd


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"out_file": File}})
def Harmonize(
    in_file: File = attrs.NOTHING,
    wm_mask: File = attrs.NOTHING,
    erodemsk: bool = True,
    thresh: float = 0.9,
) -> File:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.harmonize import Harmonize

    """
    out_file = attrs.NOTHING
    in_file = nb.load(in_file)
    wm_mask = nb.load(wm_mask).get_fdata()
    wm_mask[wm_mask < 0.9] = 0
    wm_mask[wm_mask > 0] = 1
    wm_mask = wm_mask.astype(np.uint8)

    if erodemsk:

        struct = nd.generate_binary_structure(3, 2)

        wm_mask = nd.binary_erosion(wm_mask, structure=struct).astype(np.uint8)

    data = in_file.get_fdata()
    data *= 1000.0 / np.median(data[wm_mask > 0])

    out_file = fname_presuffix(in_file, suffix="_harmonized", newpath=".")
    in_file.__class__(data, in_file.affine, in_file.header).to_filename(out_file)

    out_file = out_file

    return out_file


# Nipype methods converted into functions
