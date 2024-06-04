import attrs
from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
import pydra.mark


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"n_volumes_to_discard": int}})
def NonSteadyStateDetector(in_file: File = attrs.NOTHING) -> int:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.non_steady_state_detector import NonSteadyStateDetector

    """
    n_volumes_to_discard = attrs.NOTHING
    self_dict = {}
    in_nii = nb.load(in_file)
    global_signal = in_nii.dataobj[:, :, :, :50].mean(axis=0).mean(axis=0).mean(axis=0)

    self_dict["_results"] = {"n_volumes_to_discard": is_outlier(global_signal)}

    return n_volumes_to_discard


# Nipype methods converted into functions


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
