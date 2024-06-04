import logging
import numpy as np


logger = logging.getLogger(__name__)


def normalize_mc_params(params, source):
    """
    Normalize a single row of motion parameters to the SPM format.

    SPM saves motion parameters as:
        x   Right-Left          (mm)
        y   Anterior-Posterior  (mm)
        z   Superior-Inferior   (mm)
        rx  Pitch               (rad)
        ry  Roll                (rad)
        rz  Yaw                 (rad)
    """
    if source.upper() == "FSL":
        params = params[[3, 4, 5, 0, 1, 2]]
    elif source.upper() in ("AFNI", "FSFAST"):
        params = params[np.asarray([4, 5, 3, 1, 2, 0]) + (len(params) > 6)]
        params[3:] = params[3:] * np.pi / 180.0
    elif source.upper() == "NIPY":
        from nipy.algorithms.registration import aff2euler, to_matrix44

        matrix = to_matrix44(params)
        params = np.zeros(6)
        params[:3] = matrix[:3, 3]
        params[-1:2:-1] = aff2euler(matrix)
    return params
