import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
import pydra.mark


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"out_prov": dict}})
def AddProvenance(
    in_file: File = attrs.NOTHING,
    air_msk: File = attrs.NOTHING,
    rot_msk: File = attrs.NOTHING,
    modality: str = attrs.NOTHING,
) -> dict:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.reports.add_provenance import AddProvenance

    """
    out_prov = attrs.NOTHING
    from nipype.utils.filemanip import hash_infile

    out_prov = {
        "md5sum": hash_infile(in_file),
        "version": '<version-not-captured>',
        "software": "mriqc",
        "settings": {
            "testing": False,
        },
    }

    if modality in ("T1w", "T2w"):
        air_msk_size = np.asanyarray(nb.load(air_msk).dataobj).astype(bool).sum()
        rot_msk_size = np.asanyarray(nb.load(rot_msk).dataobj).astype(bool).sum()
        out_prov["warnings"] = {
            "small_air_mask": bool(air_msk_size < 5e5),
            "large_rot_frame": bool(rot_msk_size > 500),
        }

    if modality == "bold":
        out_prov["settings"].update(
            {
                "fd_thres": 0.2,  # <configuration>.fd_thres
            }
        )

    return out_prov


# Nipype methods converted into functions
