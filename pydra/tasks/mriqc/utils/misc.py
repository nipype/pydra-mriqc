from collections import OrderedDict
from collections.abc import Iterable
import logging


logger = logging.getLogger(__name__)


def _datalad_get(
    input_list,
    nprocs=None,
    nipype_nprocs=12,
    exec_bids_dir=None,
    exec_bids_dir_datalad=False,
    nipype_omp_nthreads=12,
):

    from pydra.tasks.mriqc import config

    if not exec_bids_dir_datalad:
        return
    # Delay datalad import until we're sure we'll need it
    import logging
    from datalad.api import get

    _dataladlog = logging.getLogger("datalad")
    _dataladlog.setLevel(logging.WARNING)
    logger.log(
        25, "DataLad dataset identified, attempting to `datalad get` unavailable files."
    )
    return get(
        list(_flatten_list(input_list)),
        dataset=str(exec_bids_dir),
        jobs=(
            nprocs
            if not None
            else max(
                nipype_omp_nthreads,
                nipype_nprocs,
            )
        ),
    )


def _flatten_dict(indict):

    out_qc = {}
    for k, value in list(indict.items()):
        if not isinstance(value, dict):
            out_qc[k] = value
        else:
            for subk, subval in list(value.items()):
                if not isinstance(subval, dict):
                    out_qc["_".join([k, subk])] = subval
                else:
                    for ssubk, ssubval in list(subval.items()):
                        out_qc["_".join([k, subk, ssubk])] = ssubval
    return out_qc


def _flatten_list(xs):

    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten_list(x)
        else:
            yield x


BIDS_COMP = OrderedDict(
    [
        ("subject_id", "sub"),
        ("session_id", "ses"),
        ("task_id", "task"),
        ("acq_id", "acq"),
        ("rec_id", "rec"),
        ("run_id", "run"),
    ]
)
