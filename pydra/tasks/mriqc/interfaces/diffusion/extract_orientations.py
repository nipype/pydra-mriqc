import attrs
from fileformats.generic import File
import logging
import nibabel as nb
from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
import numpy as np
import os
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"out_file": File, "out_bvec": list}})
def ExtractOrientations(
    in_file: File = attrs.NOTHING,
    indices: list = attrs.NOTHING,
    in_bvec_file: File = attrs.NOTHING,
) -> ty.Tuple[File, list]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.extract_orientations import ExtractOrientations

    """
    out_file = attrs.NOTHING
    out_bvec = attrs.NOTHING
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(
        in_file,
        suffix="_subset",
        newpath=os.getcwd(),
    )

    out_file = out_file

    img = nb.load(in_file)
    bzeros = np.squeeze(np.asanyarray(img.dataobj)[..., indices])

    hdr = img.header.copy()
    hdr.set_data_shape(bzeros.shape)
    hdr.set_xyzt_units("mm")
    nb.Nifti1Image(bzeros, img.affine, hdr).to_filename(out_file)

    if in_bvec_file is not attrs.NOTHING:
        bvecs = np.loadtxt(in_bvec_file)[:, indices].T
        out_bvec = [tuple(row) for row in bvecs]

    return out_file, out_bvec


# Nipype methods converted into functions
