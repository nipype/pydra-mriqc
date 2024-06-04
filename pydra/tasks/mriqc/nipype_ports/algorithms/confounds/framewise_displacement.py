import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.nipype_ports.utils.misc import normalize_mc_params
import numpy as np
import os.path as op
from pathlib import Path
import pydra.mark
from pydra.tasks.mriqc.nipype_ports.utils.misc import normalize_mc_params
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {"return": {"out_file": File, "out_figure": File, "fd_average": float}}
)
def FramewiseDisplacement(
    in_file: File = attrs.NOTHING,
    parameter_source: ty.Any = attrs.NOTHING,
    radius: float = 50,
    out_file: Path = "fd_power_2012.txt",
    out_figure: Path = "fd_power_2012.pdf",
    series_tr: float = attrs.NOTHING,
    save_plot: bool = False,
    normalize: bool = False,
    figdpi: int = 100,
    figsize: ty.Any = (11.7, 2.3),
) -> ty.Tuple[File, File, float]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.nipype_ports.algorithms.confounds.framewise_displacement import FramewiseDisplacement

    """
    out_file = attrs.NOTHING
    out_figure = attrs.NOTHING
    fd_average = attrs.NOTHING
    self_dict = {}
    mpars = np.loadtxt(in_file)  # mpars is N_t x 6
    mpars = np.apply_along_axis(
        func1d=normalize_mc_params,
        axis=1,
        arr=mpars,
        source=parameter_source,
    )
    diff = mpars[:-1, :6] - mpars[1:, :6]
    diff[:, 3:6] *= radius
    fd_res = np.abs(diff).sum(axis=1)

    self_dict["_results"] = {
        "out_file": op.abspath(out_file),
        "fd_average": float(fd_res.mean()),
    }
    np.savetxt(out_file, fd_res, header="FramewiseDisplacement", comments="")

    if save_plot:
        tr = None
        if series_tr is not attrs.NOTHING:
            tr = series_tr

        if normalize and tr is None:
            IFLOGGER.warning("FD plot cannot be normalized if TR is not set")

        out_figure = op.abspath(out_figure)
        fig = plot_confound(
            fd_res,
            figsize,
            "FD",
            units="mm",
            series_tr=tr,
            normalize=normalize,
        )
        fig.savefig(
            out_figure,
            dpi=float(figdpi),
            format=out_figure[-3:],
            bbox_inches="tight",
        )
        fig.clf()

    return out_file, out_figure, fd_average


# Nipype methods converted into functions


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


IFLOGGER = logging.getLogger("nipype.interface")
