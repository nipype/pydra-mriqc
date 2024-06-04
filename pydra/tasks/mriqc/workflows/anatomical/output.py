import attrs
import logging
from pathlib import Path
from pydra.engine import Workflow
import typing as ty


logger = logging.getLogger(__name__)


def init_anat_report_wf(
    airmask=attrs.NOTHING,
    artmask=attrs.NOTHING,
    brainmask=attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    headmask=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    name: str = "anat_report_wf",
    segmentation=attrs.NOTHING,
    wf_species="human",
):
    """
    Generate the components of the individual report.

    .. workflow::

        from mriqc.workflows.anatomical.output import init_anat_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_anat_report_wf()

    """
    from pydra.tasks.nireports.interfaces import PlotMosaic

    # from mriqc.interfaces.reports import IndividualReport
    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    verbose = exec_verbose_reports
    reportlets_dir = exec_work_dir / "reportlets"
    workflow = Workflow(
        name=name,
        input_spec={
            "airmask": ty.Any,
            "artmask": ty.Any,
            "brainmask": ty.Any,
            "headmask": ty.Any,
            "in_ras": ty.Any,
            "segmentation": ty.Any,
        },
        output_spec={
            "airmask_report": ty.Any,
            "artmask_report": ty.Any,
            "bg_report": ty.Any,
            "bmask_report": ty.Any,
            "headmask_report": ty.Any,
            "segm_report": ty.Any,
            "zoom_report": ty.Any,
        },
        airmask=airmask,
        artmask=artmask,
        brainmask=brainmask,
        headmask=headmask,
        in_ras=in_ras,
        segmentation=segmentation,
    )

    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            bbox_mask_file=workflow.lzin.brainmask,
            in_file=workflow.lzin.in_ras,
            name="mosaic_zoom",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="viridis_r",
            only_noise=True,
            in_file=workflow.lzin.in_ras,
            name="mosaic_noise",
        )
    )
    if wf_species.lower() in ("rat", "mouse"):
        workflow.mosaic_zoom.inputs.view = ["coronal", "axial"]
        workflow.mosaic_noise.inputs.view = ["coronal", "axial"]

    # fmt: off
    workflow.set_output([('zoom_report', workflow.mosaic_zoom.lzout.out_file)])
    workflow.set_output([('bg_report', workflow.mosaic_noise.lzout.out_file)])
    # fmt: on

    from pydra.tasks.nireports.interfaces import PlotContours

    display_mode = "y" if wf_species.lower() in ("rat", "mouse") else "z"
    workflow.add(
        PlotContours(
            colors=["r", "g", "b"],
            cut_coords=10,
            display_mode=display_mode,
            levels=[0.5, 1.5, 2.5],
            in_contours=workflow.lzin.segmentation,
            in_file=workflow.lzin.in_ras,
            name="plot_segm",
        )
    )

    workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=10,
            display_mode=display_mode,
            levels=[0.5],
            out_file="bmask",
            in_contours=workflow.lzin.brainmask,
            in_file=workflow.lzin.in_ras,
            name="plot_bmask",
        )
    )

    workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=10,
            display_mode=display_mode,
            levels=[0.5],
            out_file="artmask",
            saturate=True,
            in_contours=workflow.lzin.artmask,
            in_file=workflow.lzin.in_ras,
            name="plot_artmask",
        )
    )

    # NOTE: humans switch on these two to coronal view.
    display_mode = "y" if wf_species.lower() in ("rat", "mouse") else "x"
    workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=6,
            display_mode=display_mode,
            levels=[0.5],
            out_file="airmask",
            in_contours=workflow.lzin.airmask,
            in_file=workflow.lzin.in_ras,
            name="plot_airmask",
        )
    )

    workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=6,
            display_mode=display_mode,
            levels=[0.5],
            out_file="headmask",
            in_contours=workflow.lzin.headmask,
            in_file=workflow.lzin.in_ras,
            name="plot_headmask",
        )
    )

    # fmt: off
    workflow.set_output([('bmask_report', workflow.plot_bmask.lzout.out_file)])
    workflow.set_output([('segm_report', workflow.plot_segm.lzout.out_file)])
    workflow.set_output([('artmask_report', workflow.plot_artmask.lzout.out_file)])
    workflow.set_output([('headmask_report', workflow.plot_headmask.lzout.out_file)])
    workflow.set_output([('airmask_report', workflow.plot_airmask.lzout.out_file)])
    # fmt: on

    return workflow
