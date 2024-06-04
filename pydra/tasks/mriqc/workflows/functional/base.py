import attrs
import logging
from pydra.tasks.mriqc.workflows.functional.output import init_func_report_wf
from pydra.tasks.niworkflows.utils.connections import pop_file as _pop
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, SpecInfo
from pydra.engine.task import FunctionTask
import pydra.mark
from pydra.tasks.niworkflows.utils.connections import pop_file as _pop
import typing as ty


logger = logging.getLogger(__name__)


def fmri_bmsk_workflow(in_file=attrs.NOTHING, name="fMRIBrainMask"):
    """
    Compute a brain mask for the input :abbr:`fMRI (functional MRI)` dataset.

    .. workflow::

        from mriqc.workflows.functional.base import fmri_bmsk_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_bmsk_workflow()


    """
    from pydra.tasks.afni.auto import Automask

    workflow = Workflow(
        name=name,
        input_spec={"in_file": ty.Any},
        output_spec={"out_file": ty.Any},
        in_file=in_file,
    )

    workflow.add(
        Automask(outputtype="NIFTI_GZ", in_file=workflow.lzin.in_file, name="afni_msk")
    )
    # Connect brain mask extraction
    # fmt: off
    workflow.set_output([('out_file', workflow.afni_msk.lzout.out_file)])
    # fmt: on

    return workflow


def epi_mni_align(
    epi_mask=attrs.NOTHING,
    epi_mean=attrs.NOTHING,
    exec_ants_float=False,
    exec_debug=False,
    name="SpatialNormalization",
    nipype_nprocs=12,
    nipype_omp_nthreads=12,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    Estimate the transform that maps the EPI space into MNI152NLin2009cAsym.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.functional.base import epi_mni_align
        from mriqc.testing import mock_config
        with mock_config():
            wf = epi_mni_align()

    """
    from pydra.tasks.ants.auto import ApplyTransforms, N4BiasFieldCorrection
    from pydra.tasks.niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )
    from templateflow.api import get as get_template

    # Get settings
    testing = exec_debug
    n_procs = nipype_nprocs
    ants_nthreads = nipype_omp_nthreads
    workflow = Workflow(
        name=name,
        input_spec={"epi_mask": ty.Any, "epi_mean": ty.Any},
        output_spec={"epi_mni": ty.Any, "epi_parc": ty.Any, "report": ty.Any},
        epi_mask=epi_mask,
        epi_mean=epi_mean,
    )

    workflow.add(
        N4BiasFieldCorrection(
            copy_header=True,
            dimension=3,
            input_image=workflow.lzin.epi_mean,
            name="n4itk",
        )
    )
    workflow.add(
        RobustMNINormalization(
            explicit_masking=False,
            flavor="testing" if testing else "precise",
            float=exec_ants_float,
            generate_report=True,
            moving="boldref",
            num_threads=ants_nthreads,
            reference="boldref",
            template=wf_template_id,
            moving_image=workflow.n4itk.lzout.output_image,
            name="norm",
        )
    )
    if wf_species.lower() == "human":
        workflow.norm.inputs.reference_image = str(
            get_template(wf_template_id, resolution=2, suffix="boldref")
        )
        workflow.norm.inputs.reference_mask = str(
            get_template(
                wf_template_id,
                resolution=2,
                desc="brain",
                suffix="mask",
            )
        )
    # adapt some population-specific settings
    else:
        from nirodents.workflows.brainextraction import _bspline_grid

        workflow.n4itk.inputs.shrink_factor = 1
        workflow.n4itk.inputs.n_iterations = [50] * 4
        workflow.norm.inputs.reference_image = str(
            get_template(wf_template_id, suffix="T2w")
        )
        workflow.norm.inputs.reference_mask = str(
            get_template(
                wf_template_id,
                desc="brain",
                suffix="mask",
            )[0]
        )
        workflow.add(FunctionTask(func=_bspline_grid, name="bspline_grid"))
        # fmt: off
        workflow.bspline_grid.inputs.in_file = workflow.lzin.epi_mean
        workflow.n4itk.inputs.args = workflow.bspline_grid.lzout.out
        # fmt: on
    # Warp segmentation into EPI space
    workflow.add(
        ApplyTransforms(
            default_value=0,
            dimension=3,
            float=True,
            interpolation="MultiLabel",
            reference_image=workflow.lzin.epi_mean,
            transforms=workflow.norm.lzout.inverse_composite_transform,
            name="invt",
        )
    )
    if wf_species.lower() == "human":
        workflow.invt.inputs.input_image = str(
            get_template(
                wf_template_id,
                resolution=1,
                desc="carpet",
                suffix="dseg",
            )
        )
    else:
        workflow.invt.inputs.input_image = str(
            get_template(
                wf_template_id,
                suffix="dseg",
            )[-1]
        )
    # fmt: off
    workflow.set_output([('epi_parc', workflow.invt.lzout.output_image)])
    workflow.set_output([('epi_mni', workflow.norm.lzout.warped_image)])
    workflow.set_output([('report', workflow.norm.lzout.out_report)])
    # fmt: on
    if wf_species.lower() == "human":
        workflow.norm.inputs.moving_mask = workflow.lzin.epi_mask

    return workflow


def hmc(
    fd_radius=attrs.NOTHING,
    in_file=attrs.NOTHING,
    name="fMRI_HMC",
    omp_nthreads=None,
    wf_biggest_file_gb=1,
    wf_deoblique=False,
    wf_despike=False,
):
    """
    Create a :abbr:`HMC (head motion correction)` workflow for fMRI.

    .. workflow::

        from mriqc.workflows.functional.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        FramewiseDisplacement,
    )
    from pydra.tasks.afni.auto import Despike, Refit, Volreg

    mem_gb = wf_biggest_file_gb
    workflow = Workflow(
        name=name,
        input_spec={"fd_radius": ty.Any, "in_file": ty.Any},
        output_spec={"mpars": ty.Any, "out_fd": ty.Any, "out_file": ty.Any},
        fd_radius=fd_radius,
        in_file=in_file,
    )

    # calculate hmc parameters
    workflow.add(
        Volreg(
            args="-Fourier -twopass", outputtype="NIFTI_GZ", zpad=4, name="estimate_hm"
        )
    )
    # Compute the frame-wise displacement
    workflow.add(
        FramewiseDisplacement(
            normalize=False,
            parameter_source="AFNI",
            in_file=workflow.estimate_hm.lzout.oned_file,
            radius=workflow.lzin.fd_radius,
            name="fdnode",
        )
    )
    # Apply transforms to other echos
    workflow.add(
        FunctionTask(
            func=_apply_transforms,
            input_spec=SpecInfo(
                name="FunctionIn",
                bases=(BaseSpec,),
                fields=[("in_file", ty.Any), ("in_xfm", ty.Any)],
            ),
            in_xfm=workflow.estimate_hm.lzout.oned_matrix_save,
            name="apply_hmc",
        )
    )
    # fmt: off
    workflow.set_output([('out_file', workflow.apply_hmc.lzout.out)])
    workflow.set_output([('mpars', workflow.estimate_hm.lzout.oned_file)])
    workflow.set_output([('out_fd', workflow.fdnode.lzout.out_file)])
    # fmt: on
    if not (wf_despike or wf_deoblique):
        # fmt: off
        workflow.estimate_hm.inputs.in_file = workflow.lzin.in_file
        workflow.apply_hmc.inputs.in_file = workflow.lzin.in_file
        # fmt: on
        return workflow
    # despiking, and deoblique
    workflow.add(Refit(deoblique=True, name="deoblique_node"))
    workflow.add(Despike(outputtype="NIFTI_GZ", name="despike_node"))
    if wf_despike and wf_deoblique:
        # fmt: off
        workflow.despike_node.inputs.in_file = workflow.lzin.in_file
        workflow.deoblique_node.inputs.in_file = workflow.despike_node.lzout.out_file

        @pydra.mark.task
        def deoblique_node_out_file_to_estimate_hm_in_file_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        workflow.add(deoblique_node_out_file_to_estimate_hm_in_file_callable(in_=workflow.deoblique_node.lzout.out_file, name="deoblique_node_out_file_to_estimate_hm_in_file_callable"))

        workflow.estimate_hm.inputs.in_file = workflow.deoblique_node_out_file_to_estimate_hm_in_file_callable.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.deoblique_node.lzout.out_file
        # fmt: on
    elif wf_despike:
        # fmt: off
        workflow.despike_node.inputs.in_file = workflow.lzin.in_file

        @pydra.mark.task
        def despike_node_out_file_to_estimate_hm_in_file_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        workflow.add(despike_node_out_file_to_estimate_hm_in_file_callable(in_=workflow.despike_node.lzout.out_file, name="despike_node_out_file_to_estimate_hm_in_file_callable"))

        workflow.estimate_hm.inputs.in_file = workflow.despike_node_out_file_to_estimate_hm_in_file_callable.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.despike_node.lzout.out_file
        # fmt: on
    elif wf_deoblique:
        # fmt: off
        workflow.deoblique_node.inputs.in_file = workflow.lzin.in_file

        @pydra.mark.task
        def deoblique_node_out_file_to_estimate_hm_in_file_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        workflow.add(deoblique_node_out_file_to_estimate_hm_in_file_callable(in_=workflow.deoblique_node.lzout.out_file, name="deoblique_node_out_file_to_estimate_hm_in_file_callable"))

        workflow.estimate_hm.inputs.in_file = workflow.deoblique_node_out_file_to_estimate_hm_in_file_callable.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.deoblique_node.lzout.out_file
        # fmt: on
    else:
        raise NotImplementedError

    return workflow


def _apply_transforms(in_file, in_xfm):

    from pathlib import Path
    from nitransforms.linear import load
    from pydra.tasks.mriqc.utils.bids import derive_bids_fname

    realigned = load(in_xfm, fmt="afni", reference=in_file, moving=in_file).apply(
        in_file
    )
    out_file = derive_bids_fname(
        in_file,
        entity="desc-realigned",
        newpath=Path.cwd(),
        absolute=True,
    )
    realigned.to_filename(out_file)
    return str(out_file)


def compute_iqms(
    brainmask=attrs.NOTHING,
    epi_mean=attrs.NOTHING,
    fd_thres=attrs.NOTHING,
    hmc_epi=attrs.NOTHING,
    hmc_fd=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    in_tsnr=attrs.NOTHING,
    name="ComputeIQMs",
    wf_biggest_file_gb=1,
    wf_fft_spikes_detector=False,
):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.functional.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import ComputeDVARS
    from pydra.tasks.afni.auto import OutlierCount, QualityIndex
    from pydra.tasks.mriqc.interfaces import (
        DerivativesDataSink,
        FunctionalQC,
        GatherTimeseries,
        IQMFileSink,
    )
    from pydra.tasks.mriqc.interfaces.reports import AddProvenance
    from pydra.tasks.mriqc.interfaces.transitional import GCOR
    from pydra.tasks.mriqc.workflows.utils import _tofloat, get_fwhmx

    mem_gb = wf_biggest_file_gb
    workflow = Workflow(
        name=name,
        input_spec={
            "brainmask": ty.Any,
            "epi_mean": ty.Any,
            "fd_thres": ty.Any,
            "hmc_epi": ty.Any,
            "hmc_fd": ty.Any,
            "in_ras": ty.Any,
            "in_tsnr": ty.Any,
        },
        output_spec={
            "dvars": ty.Any,
            "fft": ty.Any,
            "out_file": ty.Any,
            "outliers": ty.Any,
            "spikes": ty.Any,
            "spikes_num": int,
        },
        brainmask=brainmask,
        epi_mean=epi_mean,
        fd_thres=fd_thres,
        hmc_epi=hmc_epi,
        hmc_fd=hmc_fd,
        in_ras=in_ras,
        in_tsnr=in_tsnr,
    )

    # Set FD threshold

    # Compute DVARS
    workflow.add(
        ComputeDVARS(
            save_all=True,
            save_plot=False,
            in_file=workflow.lzin.hmc_epi,
            in_mask=workflow.lzin.brainmask,
            name="dvnode",
        )
    )
    # AFNI quality measures
    fwhm = get_fwhmx()
    fwhm.name = "fwhm"
    fwhm.inputs.in_file = workflow.lzin.epi_mean
    fwhm.inputs.mask = workflow.lzin.brainmask
    workflow.add(fwhm)
    workflow.fwhm.inputs.acf = True  # Only AFNI >= 16
    workflow.add(
        OutlierCount(
            fraction=True,
            out_file="outliers.out",
            in_file=workflow.lzin.hmc_epi,
            mask=workflow.lzin.brainmask,
            name="outliers",
        )
    )

    workflow.add(
        FunctionalQC(
            fd_thres=workflow.lzin.fd_thres,
            in_epi=workflow.lzin.epi_mean,
            in_fd=workflow.lzin.hmc_fd,
            in_hmc=workflow.lzin.hmc_epi,
            in_mask=workflow.lzin.brainmask,
            in_tsnr=workflow.lzin.in_tsnr,
            name="measures",
        )
    )

    # fmt: off
    workflow.set_output([('dvars', workflow.dvnode.lzout.out_all)])

    @pydra.mark.task
    def fwhm_fwhm_to_measures_in_fwhm_callable(in_: ty.Any) -> ty.Any:
        return _tofloat(in_)

    workflow.add(fwhm_fwhm_to_measures_in_fwhm_callable(in_=workflow.fwhm.lzout.fwhm, name="fwhm_fwhm_to_measures_in_fwhm_callable"))

    workflow.measures.inputs.in_fwhm = workflow.fwhm_fwhm_to_measures_in_fwhm_callable.lzout.out
    workflow.set_output([('outliers', workflow.outliers.lzout.out_file)])
    # fmt: on

    # Save to JSON file

    # Save timeseries TSV file

    # fmt: off








    workflow.set_output([('out_file', workflow.measures.lzout.out_qc)])

    # fmt: on
    # FFT spikes finder
    if True:  # wf_fft_spikes_detector: - disabled to ensure all outputs are generated
        from pydra.tasks.mriqc.workflows.utils import slice_wise_fft

        workflow.add(
            FunctionTask(
                func=slice_wise_fft,
                input_spec=SpecInfo(
                    name="FunctionIn", bases=(BaseSpec,), fields=[("in_file", ty.Any)]
                ),
                output_spec=SpecInfo(
                    name="FunctionOut",
                    bases=(BaseSpec,),
                    fields=[
                        ("n_spikes", ty.Any),
                        ("out_spikes", ty.Any),
                        ("out_fft", ty.Any),
                    ],
                ),
                name="spikes_fft",
            )
        )
        # fmt: off
        workflow.spikes_fft.inputs.in_file = workflow.lzin.in_ras
        workflow.set_output([('spikes', workflow.spikes_fft.lzout.out_spikes)])
        workflow.set_output([('fft', workflow.spikes_fft.lzout.out_fft)])
        workflow.set_output([('spikes_num', workflow.spikes_fft.lzout.n_spikes)])
        # fmt: on

    return workflow


def _parse_tout(in_file):

    if isinstance(in_file, (list, tuple)):
        return (
            [_parse_tout(f) for f in in_file]
            if len(in_file) > 1
            else _parse_tout(in_file[0])
        )
    import numpy as np

    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()


def _parse_tqual(in_file):

    if isinstance(in_file, (list, tuple)):
        return (
            [_parse_tqual(f) for f in in_file]
            if len(in_file) > 1
            else _parse_tqual(in_file[0])
        )
    import numpy as np

    with open(in_file) as fin:
        lines = fin.readlines()
    return np.mean([float(line.strip()) for line in lines if not line.startswith("++")])


def fmri_qc_workflow(
    exec_ants_float=False,
    exec_datalad_get=True,
    exec_debug=False,
    exec_float32=True,
    exec_no_sub=False,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_file=attrs.NOTHING,
    metadata=attrs.NOTHING,
    name="funcMRIQC",
    nipype_nprocs=12,
    nipype_omp_nthreads=12,
    wf_biggest_file_gb=1,
    wf_deoblique=False,
    wf_despike=False,
    wf_fd_radius=50,
    wf_fft_spikes_detector=False,
    wf_inputs=None,
    wf_min_len_bold=5,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    Initialize the (f)MRIQC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional.base import fmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_qc_workflow()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        NonSteadyStateDetector,
        TSNR,
    )
    from pydra.tasks.afni.auto import TStat
    from pydra.tasks.niworkflows.interfaces.bids import ReadSidecarJSON
    from pydra.tasks.niworkflows.interfaces.header import SanitizeImage
    from pydra.tasks.mriqc.interfaces.functional import SelectEcho

    from pydra.tasks.mriqc.utils.misc import _flatten_list as flatten

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(
        name=name,
        input_spec={"in_file": ty.Any, "metadata": dict},
        output_spec={
            "ema_report": ty.Any,
            "func_report_wf_background_report": ty.Any,
            "func_report_wf_carpet_report": ty.Any,
            "func_report_wf_mean_report": ty.Any,
            "func_report_wf_spikes_report": ty.Any,
            "func_report_wf_stdev_report": ty.Any,
            "func_report_wf_zoomed_report": ty.Any,
            "iqmswf_dvars": ty.Any,
            "iqmswf_fft": ty.Any,
            "iqmswf_out_file": ty.Any,
            "iqmswf_outliers": ty.Any,
            "iqmswf_spikes": ty.Any,
            "iqmswf_spikes_num": ty.Any,
        },
        in_file=in_file,
        metadata=metadata,
    )

    mem_gb = wf_biggest_file_gb
    

        
    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation

    # Get metadata

    workflow.add(
        SelectEcho(
            in_files=workflow.lzin.in_file,
            metadata=workflow.lzin.metadata,
            name="pick_echo",
        )
    )
    workflow.add(
        NonSteadyStateDetector(
            in_file=workflow.pick_echo.lzout.out_file, name="non_steady_state_detector"
        )
    )
    workflow.add(
        SanitizeImage(
            max_32bit=exec_float32,
            in_file=workflow.lzin.in_file,
            n_volumes_to_discard=workflow.non_steady_state_detector.lzout.n_volumes_to_discard,
            name="sanitize",
        )
    )
    # Workflow --------------------------------------------------------
    # 1. HMC: head motion correct
    workflow.add(
        hmc(
            omp_nthreads=nipype_omp_nthreads,
            wf_despike=wf_despike,
            wf_deoblique=wf_deoblique,
            wf_biggest_file_gb=wf_biggest_file_gb,
            in_file=workflow.sanitize.lzout.out_file,
            name="hmcwf",
        )
    )
    # Set HMC settings
    workflow.inputs.fd_radius = wf_fd_radius
    # 2. Compute mean fmri
    workflow.add(
        TStat(
            options="-mean",
            outputtype="NIFTI_GZ",
            in_file=workflow.hmcwf.lzout.out_file,
            name="mean",
        )
    )
    # Compute TSNR using nipype implementation
    workflow.add(TSNR(in_file=workflow.hmcwf.lzout.out_file, name="tsnr"))
    # EPI to MNI registration
    workflow.add(
        epi_mni_align(
            nipype_omp_nthreads=nipype_omp_nthreads,
            nipype_nprocs=nipype_nprocs,
            exec_debug=exec_debug,
            wf_species=wf_species,
            wf_template_id=wf_template_id,
            exec_ants_float=exec_ants_float,
            name="ema",
        )
    )
    # 7. Compute IQMs
    workflow.add(
        compute_iqms(
            wf_fft_spikes_detector=wf_fft_spikes_detector,
            wf_biggest_file_gb=wf_biggest_file_gb,
            in_ras=workflow.sanitize.lzout.out_file,
            epi_mean=workflow.mean.lzout.out_file,
            hmc_epi=workflow.hmcwf.lzout.out_file,
            hmc_fd=workflow.hmcwf.lzout.out_fd,
            in_tsnr=workflow.tsnr.lzout.tsnr_file,
            name="iqmswf",
        )
    )
    # Reports
    workflow.add(
        init_func_report_wf(
            exec_work_dir=exec_work_dir,
            wf_biggest_file_gb=wf_biggest_file_gb,
            exec_verbose_reports=exec_verbose_reports,
            wf_fft_spikes_detector=wf_fft_spikes_detector,
            wf_species=wf_species,
            in_ras=workflow.sanitize.lzout.out_file,
            epi_mean=workflow.mean.lzout.out_file,
            in_stddev=workflow.tsnr.lzout.stddev_file,
            hmc_fd=workflow.hmcwf.lzout.out_fd,
            hmc_epi=workflow.hmcwf.lzout.out_file,
            epi_parc=workflow.ema.lzout.epi_parc,
            meta_sidecar=workflow.lzin.metadata,
            name="func_report_wf",
        )
    )
    # fmt: off

    @pydra.mark.task
    def mean_out_file_to_ema_epi_mean_callable(in_: ty.Any) -> ty.Any:
        return _pop(in_)

    workflow.add(mean_out_file_to_ema_epi_mean_callable(in_=workflow.mean.lzout.out_file, name="mean_out_file_to_ema_epi_mean_callable"))

    workflow.ema.inputs.epi_mean = workflow.mean_out_file_to_ema_epi_mean_callable.lzout.out

    # fmt: on
    if wf_fft_spikes_detector:
        # fmt: off
        workflow.set_output([('iqmswf_spikes', workflow.iqmswf.lzout.spikes)])
        workflow.set_output([('iqmswf_fft', workflow.iqmswf.lzout.fft)])
        # fmt: on
    # population specific changes to brain masking
    if wf_species == "human":
        from pydra.tasks.mriqc.workflows.shared import (
            synthstrip_wf as fmri_bmsk_workflow,
        )

        workflow.add(
            fmri_bmsk_workflow(omp_nthreads=nipype_omp_nthreads, name="skullstrip_epi")
        )
        # fmt: off

        @pydra.mark.task
        def mean_out_file_to_skullstrip_epi_in_files_callable(in_: ty.Any) -> ty.Any:
            return _pop(in_)

        workflow.add(mean_out_file_to_skullstrip_epi_in_files_callable(in_=workflow.mean.lzout.out_file, name="mean_out_file_to_skullstrip_epi_in_files_callable"))

        workflow.skullstrip_epi.inputs.in_files = workflow.mean_out_file_to_skullstrip_epi_in_files_callable.lzout.out
        workflow.ema.inputs.epi_mask = workflow.skullstrip_epi.lzout.out_mask
        workflow.iqmswf.inputs.brainmask = workflow.skullstrip_epi.lzout.out_mask
        workflow.func_report_wf.inputs.brainmask = workflow.skullstrip_epi.lzout.out_mask
        # fmt: on
    else:
        from pydra.tasks.mriqc.workflows.anatomical.base import _binarize

        workflow.add(
            FunctionTask(
                func=_binarize,
                input_spec=SpecInfo(
                    name="FunctionIn",
                    bases=(BaseSpec,),
                    fields=[("in_file", ty.Any), ("threshold", ty.Any)],
                ),
                output_spec=SpecInfo(
                    name="FunctionOut", bases=(BaseSpec,), fields=[("out_file", ty.Any)]
                ),
                name="binarise_labels",
            )
        )
        # fmt: off
        workflow.binarise_labels.inputs.in_file = workflow.ema.lzout.epi_parc
        workflow.iqmswf.inputs.brainmask = workflow.binarise_labels.lzout.out_file
        workflow.func_report_wf.inputs.brainmask = workflow.binarise_labels.lzout.out_file
        # fmt: on
    # Upload metrics
    if not exec_no_sub:
        from pydra.tasks.mriqc.interfaces.webapi import UploadIQMs

        pass

        # fmt: on
    workflow.set_output([("ema_report", workflow.ema.lzout.report)])
    workflow.set_output([("iqmswf_dvars", workflow.iqmswf.lzout.dvars)])
    workflow.set_output([("iqmswf_fft", workflow.iqmswf.lzout.fft)])
    workflow.set_output([("iqmswf_out_file", workflow.iqmswf.lzout.out_file)])
    workflow.set_output([("iqmswf_outliers", workflow.iqmswf.lzout.outliers)])
    workflow.set_output([("iqmswf_spikes", workflow.iqmswf.lzout.spikes)])
    workflow.set_output([("iqmswf_spikes_num", workflow.iqmswf.lzout.spikes_num)])
    workflow.set_output(
        [
            (
                "func_report_wf_background_report",
                workflow.func_report_wf.lzout.background_report,
            )
        ]
    )
    workflow.set_output(
        [("func_report_wf_spikes_report", workflow.func_report_wf.lzout.spikes_report)]
    )
    workflow.set_output(
        [("func_report_wf_carpet_report", workflow.func_report_wf.lzout.carpet_report)]
    )
    workflow.set_output(
        [("func_report_wf_mean_report", workflow.func_report_wf.lzout.mean_report)]
    )
    workflow.set_output(
        [("func_report_wf_stdev_report", workflow.func_report_wf.lzout.stdev_report)]
    )
    workflow.set_output(
        [("func_report_wf_zoomed_report", workflow.func_report_wf.lzout.zoomed_report)]
    )

    return workflow
