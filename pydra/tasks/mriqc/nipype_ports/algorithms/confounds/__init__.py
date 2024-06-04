import attrs
from fileformats.generic import Directory, File
import logging
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, MultiInputObj, SpecInfo
from pydra.engine.task import FunctionTask
import pydra.mark
from .compute_dvars import ComputeDVARS
from .framewise_displacement import FramewiseDisplacement
from .non_steady_state_detector import NonSteadyStateDetector
from .tsnr import TSNR
import typing as ty


logger = logging.getLogger(__name__)


def _AR_est_YW(x, order, rxx=None):
    """Retrieve AR coefficients while dropping the sig_sq return value"""
    from nitime.algorithms import AR_est_YW

    return AR_est_YW(x, order, rxx=rxx)[0]


def compute_dvars(
    in_file,
    in_mask,
    remove_zerovariance=False,
    intensity_normalization=1000,
    variance_tol=0.0,
):
    """
    Compute the :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Power2012]_.

    Particularly, the *standardized* :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Nichols2013]_ are computed.

    .. [Nichols2013] Nichols T, `Notes on creating a standardized version of
         DVARS <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic- research/nichols/scripts/fsl/standardizeddvars.pdf>`_, 2013.

    .. note:: Implementation details

      Uses the implementation of the `Yule-Walker equations
      from nitime
      <http://nipy.org/nitime/api/generated/nitime.algorithms.autoregressive.html #nitime.algorithms.autoregressive.AR_est_YW>`_
      for the :abbr:`AR (auto-regressive)` filtering of the fMRI signal.

    :param numpy.ndarray func: functional data, after head-motion-correction.
    :param numpy.ndarray mask: a 3D mask of the brain
    :param bool output_all: write out all dvars
    :param str out_file: a path to which the standardized dvars should be saved.
    :return: the standardized DVARS

    """
    import numpy as np
    import nibabel as nb
    import warnings

    func = np.float32(nb.load(in_file).dataobj)
    mask = np.bool_(nb.load(in_mask).dataobj)
    if len(func.shape) != 4:
        raise RuntimeError("Input fMRI dataset should be 4-dimensional")
    mfunc = func[mask]
    if intensity_normalization != 0:
        mfunc = (mfunc / np.median(mfunc)) * intensity_normalization
    # Robust standard deviation (we are using "lower" interpolation
    # because this is what FSL is doing
    try:
        func_sd = (
            np.percentile(mfunc, 75, axis=1, method="lower")
            - np.percentile(mfunc, 25, axis=1, method="lower")
        ) / 1.349
    except TypeError:  # NP < 1.22
        func_sd = (
            np.percentile(mfunc, 75, axis=1, interpolation="lower")
            - np.percentile(mfunc, 25, axis=1, interpolation="lower")
        ) / 1.349
    if remove_zerovariance:
        zero_variance_voxels = func_sd > variance_tol
        mfunc = mfunc[zero_variance_voxels, :]
        func_sd = func_sd[zero_variance_voxels]
    # Compute (non-robust) estimate of lag-1 autocorrelation
    ar1 = np.apply_along_axis(
        _AR_est_YW, 1, regress_poly(0, mfunc, remove_mean=True)[0].astype(np.float32), 1
    )
    # Compute (predicted) standard deviation of temporal difference time series
    diff_sdhat = np.squeeze(np.sqrt(((1 - ar1) * 2).tolist())) * func_sd
    diff_sd_mean = diff_sdhat.mean()
    # Compute temporal difference time series
    func_diff = np.diff(mfunc, axis=1)
    # DVARS (no standardization)
    dvars_nstd = np.sqrt(np.square(func_diff).mean(axis=0))
    # standardization
    dvars_stdz = dvars_nstd / diff_sd_mean
    with warnings.catch_warnings():  # catch, e.g., divide by zero errors
        warnings.filterwarnings("error")
        # voxelwise standardization
        diff_vx_stdz = np.square(
            func_diff / np.array([diff_sdhat] * func_diff.shape[-1]).T
        )
        dvars_vx_stdz = np.sqrt(diff_vx_stdz.mean(axis=0))
    return (dvars_stdz, dvars_nstd, dvars_vx_stdz)


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    :param nparray points: an numobservations by numdimensions numpy array of observations
    :param float thresh: the modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

    :return: A boolean mask, of size numobservations-length array.

    .. note:: References

        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    timepoints_to_discard = 0
    for i in range(len(modified_z_score)):
        if modified_z_score[i] <= thresh:
            break
        else:
            timepoints_to_discard += 1
    return timepoints_to_discard


def plot_confound(tseries, figsize, name, units=None, series_tr=None, normalize=False):
    """
    A helper function to plot :abbr:`fMRI (functional MRI)` confounds.

    """
    import matplotlib

    matplotlib.use(config.get("execution", "matplotlib_backend"))
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
    import seaborn as sns

    fig = plt.Figure(figsize=figsize)
    FigureCanvas(fig)
    grid = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.025)
    grid.update(hspace=1.0, right=0.95, left=0.1, bottom=0.2)
    ax = fig.add_subplot(grid[0, :-1])
    if normalize and series_tr is not None:
        tseries /= series_tr
    ax.plot(tseries)
    ax.set_xlim((0, len(tseries)))
    ylabel = name
    if units is not None:
        ylabel += (" speed [{}/s]" if normalize else " [{}]").format(units)
    ax.set_ylabel(ylabel)
    xlabel = "Frame #"
    if series_tr is not None:
        xlabel = "Frame # ({} sec TR)".format(series_tr)
    ax.set_xlabel(xlabel)
    ylim = ax.get_ylim()
    ax = fig.add_subplot(grid[0, -1])
    sns.distplot(tseries, vertical=True, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylim(ylim)
    ax.set_yticklabels([])
    return fig


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
