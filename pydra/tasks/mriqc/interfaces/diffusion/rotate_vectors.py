import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"out_bvec": list, "out_diff": list}})
def RotateVectors(
    in_file: File = attrs.NOTHING,
    reference: File = attrs.NOTHING,
    transforms: File = attrs.NOTHING,
) -> ty.Tuple[list, list]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.rotate_vectors import RotateVectors

    """
    out_bvec = attrs.NOTHING
    out_diff = attrs.NOTHING
    from nitransforms.linear import load

    vox2ras = nb.load(reference).affine
    ras2vox = np.linalg.inv(vox2ras)

    ijk = np.loadtxt(in_file).T
    nonzero = np.linalg.norm(ijk, axis=1) > 1e-3

    xyz = (vox2ras[:3, :3] @ ijk.T).T

    xyz_norms = np.linalg.norm(xyz, axis=1)
    xyz[nonzero] = xyz[nonzero] / xyz_norms[nonzero, np.newaxis]

    hmc_rot = load(transforms).matrix[:, :3, :3]
    ijk_rotated = (ras2vox[:3, :3] @ np.einsum("ijk,ik->ij", hmc_rot, xyz).T).T.astype(
        "float32"
    )
    ijk_rotated_norm = np.linalg.norm(ijk_rotated, axis=1)
    ijk_rotated[nonzero] = ijk_rotated[nonzero] / ijk_rotated_norm[nonzero, np.newaxis]
    ijk_rotated[~nonzero] = ijk[~nonzero]

    out_bvec = list(zip(ijk_rotated[:, 0], ijk_rotated[:, 1], ijk_rotated[:, 2]))

    diffs = np.zeros_like(ijk[:, 0])
    diffs[nonzero] = np.arccos(
        np.clip(np.einsum("ij, ij->i", ijk[nonzero], ijk_rotated[nonzero]), -1.0, 1.0)
    )
    out_diff = [round(float(v), 6) for v in diffs]

    return out_bvec, out_diff


# Nipype methods converted into functions
