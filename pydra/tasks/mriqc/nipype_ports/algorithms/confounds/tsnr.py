import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
import os.path as op
from pathlib import Path
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {
        "return": {
            "tsnr_file": File,
            "mean_file": File,
            "stddev_file": File,
            "detrended_file": File,
        }
    }
)
def TSNR(
    in_file: ty.List[File] = attrs.NOTHING,
    regress_poly: ty.Any = attrs.NOTHING,
    tsnr_file: Path = "tsnr.nii.gz",
    mean_file: Path = "mean.nii.gz",
    stddev_file: Path = "stdev.nii.gz",
    detrended_file: Path = "detrend.nii.gz",
) -> ty.Tuple[File, File, File, File]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.tsnr import TSNR

    """
    tsnr_file = attrs.NOTHING
    mean_file = attrs.NOTHING
    stddev_file = attrs.NOTHING
    detrended_file = attrs.NOTHING
    img = nb.load(in_file[0])
    header = img.header.copy()
    vollist = [nb.load(filename) for filename in in_file]
    data = np.concatenate(
        [
            vol.get_fdata(dtype=np.float32).reshape(vol.shape[:3] + (-1,))
            for vol in vollist
        ],
        axis=3,
    )
    data = np.nan_to_num(data)

    if data.dtype.kind == "i":
        header.set_data_dtype(np.float32)
        data = data.astype(np.float32)

    if regress_poly is not attrs.NOTHING:
        data = regress_poly(regress_poly, data, remove_mean=False)[0]
        img = nb.Nifti1Image(data, img.affine, header)
        nb.save(img, op.abspath(detrended_file))

    meanimg = np.mean(data, axis=3)
    stddevimg = np.std(data, axis=3)
    tsnr = np.zeros_like(meanimg)
    stddevimg_nonzero = stddevimg > 1.0e-3
    tsnr[stddevimg_nonzero] = meanimg[stddevimg_nonzero] / stddevimg[stddevimg_nonzero]
    img = nb.Nifti1Image(tsnr, img.affine, header)
    nb.save(img, op.abspath(tsnr_file))
    img = nb.Nifti1Image(meanimg, img.affine, header)
    nb.save(img, op.abspath(mean_file))
    img = nb.Nifti1Image(stddevimg, img.affine, header)
    nb.save(img, op.abspath(stddev_file))
    self_dict = {}
    outputs = {}.get()
    for k in ["tsnr_file", "mean_file", "stddev_file"]:
        outputs[k] = op.abspath(getattr(self_dict["inputs"], k))

    if regress_poly is not attrs.NOTHING:
        detrended_file = op.abspath(detrended_file)

    return tsnr_file, mean_file, stddev_file, detrended_file


# Nipype methods converted into functions


def regress_poly(degree, data, remove_mean=True, axis=-1, failure_mode="error"):
    """
    Returns data with degree polynomial regressed out.

    :param bool remove_mean: whether or not demean data (i.e. degree 0),
    :param int axis: numpy array axes along which regression is performed

    """
    IFLOGGER.debug(
        "Performing polynomial regression on data of shape %s", str(data.shape)
    )
    datashape = data.shape
    timepoints = datashape[axis]
    if datashape[0] == 0 and failure_mode != "error":
        return data, np.array([])
    # Rearrange all voxel-wise time-series in rows
    data = data.reshape((-1, timepoints))
    # Generate design matrix
    X = np.ones((timepoints, 1))  # quick way to calc degree 0
    for i in range(degree):
        polynomial_func = Legendre.basis(i + 1)
        value_array = np.linspace(-1, 1, timepoints)
        X = np.hstack((X, polynomial_func(value_array)[:, np.newaxis]))
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])
    # Calculate coefficients
    betas = np.linalg.pinv(X).dot(data.T)
    # Estimation
    if remove_mean:
        datahat = X.dot(betas).T
    else:  # disregard the first layer of X, which is degree 0
        datahat = X[:, 1:].dot(betas[1:, ...]).T
    regressed_data = data - datahat
    # Back to original shape
    return regressed_data.reshape(datashape), non_constant_regressors


IFLOGGER = logging.getLogger("nipype.interface")
