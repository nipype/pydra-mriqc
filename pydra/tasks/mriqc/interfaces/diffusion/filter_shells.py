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
@pydra.mark.annotate(
    {
        "return": {
            "out_file": File,
            "out_bvals": list,
            "out_bvec_file": File,
            "out_bval_file": File,
        }
    }
)
def FilterShells(
    in_file: File = attrs.NOTHING,
    bvals: list = attrs.NOTHING,
    bvec_file: File = attrs.NOTHING,
    b_threshold: float = 1100,
) -> ty.Tuple[File, list, File, File]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.filter_shells import FilterShells

    """
    out_file = attrs.NOTHING
    out_bvals = attrs.NOTHING
    out_bvec_file = attrs.NOTHING
    out_bval_file = attrs.NOTHING
    from nipype.utils.filemanip import fname_presuffix

    bvals = np.array(bvals)
    bval_mask = bvals < b_threshold
    bvecs = np.loadtxt(bvec_file)[:, bval_mask]

    out_bvals = bvals[bval_mask].astype(float).tolist()
    out_bvec_file = fname_presuffix(
        in_file,
        suffix="_dti.bvec",
        newpath=os.getcwd(),
        use_ext=False,
    )
    np.savetxt(out_bvec_file, bvecs)

    out_bval_file = fname_presuffix(
        in_file,
        suffix="_dti.bval",
        newpath=os.getcwd(),
        use_ext=False,
    )
    np.savetxt(out_bval_file, bvals)

    out_file = fname_presuffix(
        in_file,
        suffix="_dti",
        newpath=os.getcwd(),
    )

    dwi_img = nb.load(in_file)
    data = np.array(dwi_img.dataobj, dtype=dwi_img.header.get_data_dtype())[
        ..., bval_mask
    ]
    dwi_img.__class__(
        data,
        dwi_img.affine,
        dwi_img.header,
    ).to_filename(out_file)

    return out_file, out_bvals, out_bvec_file, out_bval_file


# Nipype methods converted into functions
