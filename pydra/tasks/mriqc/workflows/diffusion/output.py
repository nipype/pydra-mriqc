import attrs
import logging
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.task import FunctionTask
from pydra.tasks.nireports.interfaces.dmri import DWIHeatmap
from pydra.tasks.nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
import typing as ty


logger = logging.getLogger(__name__)


def init_dwi_report_wf(
    brain_mask=attrs.NOTHING,
    epi_mean=attrs.NOTHING,
    epi_parc=attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    fd_thres=attrs.NOTHING,
    hmc_epi=attrs.NOTHING,
    hmc_fd=attrs.NOTHING,
    in_avgmap=attrs.NOTHING,
    in_bdict=attrs.NOTHING,
    in_dvars=attrs.NOTHING,
    in_epi=attrs.NOTHING,
    in_fa=attrs.NOTHING,
    in_fft=attrs.NOTHING,
    in_md=attrs.NOTHING,
    in_parcellation=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    in_spikes=attrs.NOTHING,
    in_stdmap=attrs.NOTHING,
    meta_sidecar=attrs.NOTHING,
    name="dwi_report_wf",
    noise_floor=attrs.NOTHING,
    outliers=attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fd_thres=0.2,
    wf_fft_spikes_detector=False,
    wf_species="human",
):
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.diffusion.output import init_dwi_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_dwi_report_wf()

    """
    from pydra.tasks.nireports.interfaces import FMRISummary, PlotMosaic, PlotSpikes
    from pydra.tasks.niworkflows.interfaces.morphology import (
        BinaryDilation,
        BinarySubtraction,
    )

    # from mriqc.interfaces.reports import IndividualReport
    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    verbose = exec_verbose_reports
    mem_gb = wf_biggest_file_gb
    reportlets_dir = exec_work_dir / "reportlets"
    workflow = Workflow(
        name=name,
        input_spec={
            "brain_mask": ty.Any,
            "epi_mean": ty.Any,
            "epi_parc": ty.Any,
            "fd_thres": ty.Any,
            "hmc_epi": ty.Any,
            "hmc_fd": ty.Any,
            "in_avgmap": ty.Any,
            "in_bdict": ty.Any,
            "in_dvars": ty.Any,
            "in_epi": ty.Any,
            "in_fa": ty.Any,
            "in_fft": ty.Any,
            "in_md": ty.Any,
            "in_parcellation": ty.Any,
            "in_ras": ty.Any,
            "in_spikes": ty.Any,
            "in_stdmap": ty.Any,
            "meta_sidecar": ty.Any,
            "noise_floor": ty.Any,
            "outliers": ty.Any,
        },
        output_spec={
            "bmask_report": ty.Any,
            "carpet_report": ty.Any,
            "fa_report": ty.Any,
            "heatmap_report": ty.Any,
            "md_report": ty.Any,
            "noise_report": ty.Any,
            "snr_report": ty.Any,
            "spikes_report": ty.Any,
        },
        brain_mask=brain_mask,
        epi_mean=epi_mean,
        epi_parc=epi_parc,
        fd_thres=fd_thres,
        hmc_epi=hmc_epi,
        hmc_fd=hmc_fd,
        in_avgmap=in_avgmap,
        in_bdict=in_bdict,
        in_dvars=in_dvars,
        in_epi=in_epi,
        in_fa=in_fa,
        in_fft=in_fft,
        in_md=in_md,
        in_parcellation=in_parcellation,
        in_ras=in_ras,
        in_spikes=in_spikes,
        in_stdmap=in_stdmap,
        meta_sidecar=meta_sidecar,
        noise_floor=noise_floor,
        outliers=outliers,
    )

    # Set FD threshold
    # inputnode.inputs.fd_thres = wf_fd_thres
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            bbox_mask_file=workflow.lzin.brain_mask,
            in_file=workflow.lzin.in_fa,
            name="mosaic_fa",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            bbox_mask_file=workflow.lzin.brain_mask,
            in_file=workflow.lzin.in_md,
            name="mosaic_md",
        )
    )
    workflow.add(
        SimpleBeforeAfter(
            after_label="Standard Deviation",
            before_label="Average",
            dismiss_affine=True,
            fixed_params={"cmap": "viridis"},
            moving_params={"cmap": "Greys_r"},
            after=workflow.lzin.in_stdmap,
            before=workflow.lzin.in_avgmap,
            wm_seg=workflow.lzin.brain_mask,
            name="mosaic_snr",
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
        workflow.mosaic_noise.inputs.view = ["coronal", "axial"]
        workflow.mosaic_fa.inputs.view = ["coronal", "axial"]
        workflow.mosaic_md.inputs.view = ["coronal", "axial"]

    def _gen_entity(inlist):
        return ["00000"] + [f"{int(round(bval, 0)):05d}" for bval in inlist]

    # fmt: off


    workflow.set_output([('snr_report', workflow.mosaic_snr.lzout.out_report)])
    workflow.set_output([('noise_report', workflow.mosaic_noise.lzout.out_file)])
    workflow.set_output([('fa_report', workflow.mosaic_fa.lzout.out_file)])
    workflow.set_output([('md_report', workflow.mosaic_md.lzout.out_file)])
    # fmt: on
    workflow.add(
        FunctionTask(func=_get_wm, in_file=workflow.lzin.in_parcellation, name="get_wm")
    )
    workflow.add(
        DWIHeatmap(
            scalarmap_label="Shell-wise Fractional Anisotropy (FA)",
            b_indices=workflow.lzin.in_bdict,
            in_file=workflow.lzin.in_epi,
            mask_file=workflow.get_wm.lzout.out,
            scalarmap=workflow.lzin.in_fa,
            sigma=workflow.lzin.noise_floor,
            name="plot_heatmap",
        )
    )

    # fmt: off
    workflow.set_output([('heatmap_report', workflow.plot_heatmap.lzout.out_file)])
    # fmt: on

    # Generate crown mask
    # Create the crown mask
    workflow.add(BinaryDilation(in_mask=workflow.lzin.brain_mask, name="dilated_mask"))
    workflow.add(
        BinarySubtraction(
            in_base=workflow.dilated_mask.lzout.out_mask,
            in_subtract=workflow.lzin.brain_mask,
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
            outliers=workflow.lzin.outliers,
            tr=workflow.lzin.meta_sidecar,
            name="bigplot",
        )
    )

    # fmt: off
    workflow.bigplot.inputs.tr = workflow.lzin.meta_sidecar
    workflow.set_output([('carpet_report', workflow.bigplot.lzout.out_file)])
    # fmt: on
    if True:  # wf_fft_spikes_detector:
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
    if False:  # not verbose:
        return workflow
    # Verbose-reporting goes here
    from pydra.tasks.nireports.interfaces import PlotContours

    workflow.add(
        PlotContours(
            colors=["r"],
            cut_coords=10,
            display_mode="y" if wf_species.lower() in ("rat", "mouse") else "z",
            levels=[0.5],
            out_file="bmask",
            in_contours=workflow.lzin.brain_mask,
            in_file=workflow.lzin.epi_mean,
            name="plot_bmask",
        )
    )

    # fmt: off
    workflow.set_output([('bmask_report', workflow.plot_bmask.lzout.out_file)])
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

    return meta_dict.get("RepetitionTime", None)


def _get_wm(in_file, radius=2):

    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    parc = nb.load(in_file)
    hdr = parc.header.copy()
    data = np.array(parc.dataobj, dtype=hdr.get_data_dtype())
    wm_mask = ndi.binary_erosion((data == 1) | (data == 2), ball(radius))
    hdr.set_data_dtype(np.uint8)
    out_wm = fname_presuffix(in_file, suffix="wm", newpath=str(Path.cwd()))
    parc.__class__(
        wm_mask.astype(np.uint8),
        parc.affine,
        hdr,
    ).to_filename(out_wm)
    return out_wm
