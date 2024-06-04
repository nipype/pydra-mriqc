import attrs
from fileformats.generic import Directory, File
import logging
import numpy as np
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {
        "return": {
            "out_bvec_file": File,
            "out_bval_file": File,
            "out_bmatrix": list,
            "qspace_neighbors": list,
            "out_dict": dict,
            "subject": str,
            "session": str,
            "task": str,
            "acquisition": str,
            "reconstruction": str,
            "run": int,
            "suffix": str,
        }
    }
)
def ReadDWIMetadata(
    in_file: File = attrs.NOTHING,
    bids_dir: ty.Any = attrs.NOTHING,
    bids_validate: bool = True,
    index_db: Directory = attrs.NOTHING,
) -> ty.Tuple[File, File, list, list, dict, str, str, str, str, str, int, str]:
    """
    Examples
    -------

    >>> from fileformats.generic import Directory, File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.read_dwi_metadata import ReadDWIMetadata

    """
    out_bvec_file = attrs.NOTHING
    out_bval_file = attrs.NOTHING
    out_bmatrix = attrs.NOTHING
    qspace_neighbors = attrs.NOTHING
    out_dict = attrs.NOTHING
    subject = attrs.NOTHING
    session = attrs.NOTHING
    task = attrs.NOTHING
    acquisition = attrs.NOTHING
    reconstruction = attrs.NOTHING
    run = attrs.NOTHING
    suffix = attrs.NOTHING
    self_dict = {}
    from bids.utils import listify

    self_dict["_fields"] = listify(fields or [])
    self_dict["_undef_fields"] = undef_fields
    self_dict = {}
    runtime = _run_interface(runtime)

    out_bvec_file = str(self_dict["layout"].get_bvec(in_file))
    out_bval_file = str(self_dict["layout"].get_bval(in_file))

    bvecs = np.loadtxt(out_bvec_file).T
    bvals = np.loadtxt(out_bval_file)

    qspace_neighbors = _find_qspace_neighbors(bvals, bvecs)
    out_bmatrix = np.hstack((bvecs, bvals[:, np.newaxis])).tolist()

    return (
        out_bvec_file,
        out_bval_file,
        out_bmatrix,
        qspace_neighbors,
        out_dict,
        subject,
        session,
        task,
        acquisition,
        reconstruction,
        run,
        suffix,
    )


# Nipype methods converted into functions


def _find_qspace_neighbors(
    bvals: np.ndarray, bvecs: np.ndarray
) -> list[tuple[int, int]]:
    """
    Create a mapping of dwi volume index to its nearest neighbor in q-space.

    This function implements an approximate nearest neighbor search in q-space
    (excluding delta encoding). It calculates the Cartesian distance between
    q-space representations of each diffusion-weighted imaging (DWI) volume
    (represented by b-values and b-vectors) and identifies the closest one
    (excluding the volume itself and b=0 volumes).

    Parameters
    ----------
    bvals : :obj:`~numpy.ndarray`
        List of b-values.
    bvecs : :obj:`~numpy.ndarray`
        Table of b-vectors.

    Returns
    -------
    :obj:`list` of :obj:`tuple`
        A list of 2-tuples indicating the nearest q-space neighbor
        of each dwi volume.

    Examples
    --------
    >>> _find_qspace_neighbors(
    ...     np.array([0, 1000, 1000, 2000]),
    ...     np.array([
    ...         [1, 0, 0],
    ...         [1, 0, 0],
    ...         [0.99, 0.0001, 0.0001],
    ...         [1, 0, 0]
    ...     ]),
    ... )
    [(1, 2), (2, 1), (3, 1)]

    Notes
    -----
    This is a copy of DIPY's code to be removed (and just imported) as soon as
    a new release of DIPY is cut including
    `dipy/dipy#3156 <https://github.com/dipy/dipy/pull/3156>`__.

    """
    from dipy.core.geometry import cart_distance
    from dipy.core.gradients import gradient_table

    gtab = gradient_table(bvals, bvecs)
    dwi_neighbors: list[tuple[int, int]] = []
    # Only correlate the b>0 images
    dwi_indices = np.flatnonzero(~gtab.b0s_mask)
    # Get a pseudo-qspace value for b>0s
    qvecs = np.sqrt(gtab.bvals)[:, np.newaxis] * gtab.bvecs
    for dwi_index in dwi_indices:
        qvec = qvecs[dwi_index]
        # Calculate distance in q-space, accounting for symmetry
        pos_dist = cart_distance(qvec[np.newaxis, :], qvecs)
        neg_dist = cart_distance(qvec[np.newaxis, :], -qvecs)
        distances = np.min(np.column_stack([pos_dist, neg_dist]), axis=1)
        # Be sure we don't select the image as its own neighbor
        distances[dwi_index] = np.inf
        # Or a b=0
        distances[gtab.b0s_mask] = np.inf
        neighbor_index = np.argmin(distances)
        dwi_neighbors.append((dwi_index, neighbor_index))
    return dwi_neighbors
