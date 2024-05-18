import attrs
import logging
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, SpecInfo
from pydra.engine.task import FunctionTask
import typing as ty


logger = logging.getLogger(__name__)


def init_func_report_wf(
    brainmask=attrs.NOTHING,
    epi_mean=attrs.NOTHING,
    epi_parc=attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    fd_thres=attrs.NOTHING,
    hmc_epi=attrs.NOTHING,
    hmc_fd=attrs.NOTHING,
    in_dvars=attrs.NOTHING,
    in_fft=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    in_spikes=attrs.NOTHING,
    in_stddev=attrs.NOTHING,
    meta_sidecar=attrs.NOTHING,
    name="func_report_wf",
    outliers=attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fft_spikes_detector=False,
    wf_species="human",
):
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.functional.output import init_func_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_func_report_wf()

    """
    from pydra.tasks.nireports.interfaces import FMRISummary, PlotMosaic, PlotSpikes
    from pydra.tasks.niworkflows.interfaces.morphology import (
        BinaryDilation,
        BinarySubtraction,
    )
    from pydra.tasks.mriqc.interfaces.functional import Spikes

    # from mriqc.interfaces.reports import IndividualReport
    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    verbose = exec_verbose_reports
    mem_gb = wf_biggest_file_gb
    reportlets_dir = exec_work_dir / "reportlets"
    workflow = Workflow(
        name=name,
        input_spec={
            "brainmask": ty.Any,
            "epi_mean": ty.Any,
            "epi_parc": ty.Any,
            "fd_thres": ty.Any,
            "hmc_epi": ty.Any,
            "hmc_fd": ty.Any,
            "in_dvars": ty.Any,
            "in_fft": ty.Any,
            "in_ras": ty.Any,
            "in_spikes": ty.Any,
            "in_stddev": ty.Any,
            "meta_sidecar": ty.Any,
            "outliers": ty.Any,
        },
        output_spec={
            "background_report": ty.Any,
            "carpet_report": ty.Any,
            "mean_report": ty.Any,
            "spikes_report": ty.Any,
            "stdev_report": ty.Any,
            "zoomed_report": ty.Any,
        },
        brainmask=brainmask,
        epi_mean=epi_mean,
        epi_parc=epi_parc,
        fd_thres=fd_thres,
        hmc_epi=hmc_epi,
        hmc_fd=hmc_fd,
        in_dvars=in_dvars,
        in_fft=in_fft,
        in_ras=in_ras,
        in_spikes=in_spikes,
        in_stddev=in_stddev,
        meta_sidecar=meta_sidecar,
        outliers=outliers,
    )

    # Set FD threshold

    workflow.add(
        FunctionTask(
            func=spikes_mask,
            input_spec=SpecInfo(
                name="FunctionIn",
                bases=(BaseSpec,),
                fields=[("in_file", ty.Any), ("in_mask", ty.Any)],
            ),
            output_spec=SpecInfo(
                name="FunctionOut",
                bases=(BaseSpec,),
                fields=[("out_file", ty.Any), ("out_plot", ty.Any)],
            ),
            in_file=workflow.lzin.in_ras,
            name="spmask",
        )
    )
    workflow.add(
        Spikes(
            detrend=False,
            no_zscore=True,
            in_file=workflow.lzin.in_ras,
            in_mask=workflow.spmask.lzout.out_file,
            name="spikes_bg",
        )
    )
    # Generate crown mask
    # Create the crown mask
    workflow.add(BinaryDilation(in_mask=workflow.lzin.brainmask, name="dilated_mask"))
    workflow.add(
        BinarySubtraction(
            in_base=workflow.dilated_mask.lzout.out_mask,
            in_subtract=workflow.lzin.brainmask,
            name="subtract_mask",
        )
    )
    workflow.add(
        FunctionTask(
            func=_carpet_parcellation,
            crown_mask=workflow.subtract_mask.lzout.out_mask,
            segmentation=workflow.lzin.epi_parc,
            name="parcels",
        )
    )
    workflow.add(
        FMRISummary(
            dvars=workflow.lzin.in_dvars,
            fd=workflow.lzin.hmc_fd,
            fd_thres=workflow.lzin.fd_thres,
            in_func=workflow.lzin.hmc_epi,
            in_segm=workflow.parcels.lzout.out,
            in_spikes_bg=workflow.spikes_bg.lzout.out_tsz,
            outliers=workflow.lzin.outliers,
            tr=workflow.lzin.meta_sidecar,
            name="bigplot",
        )
    )
    # fmt: off
    workflow.bigplot.inputs.tr = workflow.lzin.meta_sidecar
    # fmt: on
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            out_file="plot_func_mean_mosaic1.svg",
            in_file=workflow.lzin.epi_mean,
            name="mosaic_mean",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="viridis",
            out_file="plot_func_stddev_mosaic2_stddev.svg",
            in_file=workflow.lzin.in_stddev,
            name="mosaic_stddev",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            bbox_mask_file=workflow.lzin.brainmask,
            in_file=workflow.lzin.epi_mean,
            name="mosaic_zoom",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="viridis_r",
            only_noise=True,
            in_file=workflow.lzin.epi_mean,
            name="mosaic_noise",
        )
    )
    if wf_species.lower() in ("rat", "mouse"):
        workflow.mosaic_mean.inputs.view = ["coronal", "axial"]
        workflow.mosaic_stddev.inputs.view = ["coronal", "axial"]
        workflow.mosaic_zoom.inputs.view = ["coronal", "axial"]
        workflow.mosaic_noise.inputs.view = ["coronal", "axial"]

    # fmt: off
    workflow.set_output([('mean_report', workflow.mosaic_mean.lzout.out_file)])
    workflow.set_output([('stdev_report', workflow.mosaic_stddev.lzout.out_file)])
    workflow.set_output([('background_report', workflow.mosaic_noise.lzout.out_file)])
    workflow.set_output([('zoomed_report', workflow.mosaic_zoom.lzout.out_file)])
    workflow.set_output([('carpet_report', workflow.bigplot.lzout.out_file)])
    # fmt: on
    if True:  # wf_fft_spikes_detector: - disabled so output is always created
        workflow.add(
            PlotSpikes(
                cmap="viridis",
                out_file="plot_spikes.svg",
                title="High-Frequency spikes",
                name="mosaic_spikes",
            )
        )
        pass
        # fmt: off
        pass
        workflow.mosaic_spikes.inputs.in_file = workflow.lzin.in_ras
        workflow.mosaic_spikes.inputs.in_spikes = workflow.lzin.in_spikes
        workflow.mosaic_spikes.inputs.in_fft = workflow.lzin.in_fft
        workflow.set_output([('spikes_report', workflow.mosaic_spikes.lzout.out_file)])
        # fmt: on
    if not verbose:
        return workflow
    # Verbose-reporting goes here
    from pydra.tasks.nireports.interfaces import PlotContours
    from pydra.tasks.niworkflows.utils.connections import pop_file as _pop

    # fmt: off

    # fmt: on

    return workflow


def _carpet_parcellation(segmentation, crown_mask):
    """Generate the union of two masks."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)
    lut = np.zeros((256,), dtype="uint8")
    lut[100:201] = 1  # Ctx GM
    lut[30:99] = 2  # dGM
    lut[1:11] = 3  # WM+CSF
    lut[255] = 4  # Cerebellum
    # Apply lookup table
    seg = lut[np.asanyarray(img.dataobj, dtype="uint16")]
    seg[np.asanyarray(nb.load(crown_mask).dataobj, dtype=int) > 0] = 5
    outimg = img.__class__(seg.astype("uint8"), img.affine, img.header)
    outimg.set_data_dtype("uint8")
    out_file = Path("segments.nii.gz").absolute()
    outimg.to_filename(out_file)
    return str(out_file)


def _get_tr(meta_dict):

    if isinstance(meta_dict, (list, tuple)):
        meta_dict = meta_dict[0]
    return meta_dict.get("RepetitionTime", None)


def spikes_mask(in_file, in_mask=None, out_file=None):
    """Calculate a mask in which check for :abbr:`EM (electromagnetic)` spikes."""
    import os.path as op
    import nibabel as nb
    import numpy as np
    from nilearn.image import mean_img
    from nilearn.plotting import plot_roi
    from scipy import ndimage as nd

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_spmask{ext}")
        out_plot = op.abspath(f"{fname}_spmask.pdf")
    in_4d_nii = nb.load(in_file)
    orientation = nb.aff2axcodes(in_4d_nii.affine)
    if in_mask:
        mask_data = np.asanyarray(nb.load(in_mask).dataobj)
        a = np.where(mask_data != 0)
        bbox = (
            np.max(a[0]) - np.min(a[0]),
            np.max(a[1]) - np.min(a[1]),
            np.max(a[2]) - np.min(a[2]),
        )
        longest_axis = np.argmax(bbox)
        # Input here is a binarized and intersected mask data from previous section
        dil_mask = nd.binary_dilation(
            mask_data, iterations=int(mask_data.shape[longest_axis] / 9)
        )
        rep = list(mask_data.shape)
        rep[longest_axis] = -1
        new_mask_2d = dil_mask.max(axis=longest_axis).reshape(rep)
        rep = [1, 1, 1]
        rep[longest_axis] = mask_data.shape[longest_axis]
        new_mask_3d = np.logical_not(np.tile(new_mask_2d, rep))
    else:
        new_mask_3d = np.zeros(in_4d_nii.shape[:3]) == 1
    if orientation[0] in ("L", "R"):
        new_mask_3d[0:2, :, :] = True
        new_mask_3d[-3:-1, :, :] = True
    else:
        new_mask_3d[:, 0:2, :] = True
        new_mask_3d[:, -3:-1, :] = True
    mask_nii = nb.Nifti1Image(
        new_mask_3d.astype(np.uint8), in_4d_nii.affine, in_4d_nii.header
    )
    mask_nii.to_filename(out_file)
    plot_roi(mask_nii, mean_img(in_4d_nii), output_file=out_plot)
    return out_file, out_plot
