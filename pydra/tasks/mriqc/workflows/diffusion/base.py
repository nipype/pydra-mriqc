import attrs
from fileformats.medimage import Bval, Bvec
import logging
import numpy as np
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.task import FunctionTask
import pydra.mark
from pydra.tasks.mriqc.workflows.diffusion.output import init_dwi_report_wf
import typing as ty


logger = logging.getLogger(__name__)


def dmri_qc_workflow(
    bvals=attrs.NOTHING,
    bvecs=attrs.NOTHING,
    exec_ants_float=False,
    exec_datalad_get=True,
    exec_debug=False,
    exec_float32=True,
    exec_layout=None,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_file=attrs.NOTHING,
    name="dwiMRIQC",
    nipype_nprocs=12,
    nipype_omp_nthreads=12,
    qspace_neighbors=attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fd_radius=50,
    wf_fd_thres=0.2,
    wf_fft_spikes_detector=False,
    wf_inputs=None,
    wf_min_len_dwi=7,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    Initialize the dMRI-QC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.diffusion.base import dmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = dmri_qc_workflow()

    """
    from pydra.tasks.afni.auto import Volreg
    from pydra.tasks.mrtrix3.v3_0 import DwiDenoise
    from pydra.tasks.niworkflows.interfaces.header import SanitizeImage
    from pydra.tasks.niworkflows.interfaces.images import RobustAverage
    from pydra.tasks.mriqc.interfaces.diffusion import (
        CCSegmentation,
        CorrectSignalDrift,
        DiffusionModel,
        ExtractOrientations,
        NumberOfShells,
        PIESNO,
        ReadDWIMetadata,
        SpikingVoxelsMask,
        WeightedStat,
    )

    from pydra.tasks.mriqc.workflows.shared import synthstrip_wf as dmri_bmsk_workflow

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(
        name=name,
        input_spec={
            "bvals": Bval,
            "bvecs": Bvec,
            "in_file": ty.Any,
            "qspace_neighbors": ty.Any,
        },
        output_spec={
            "dwi_report_wf_bmask_report": ty.Any,
            "dwi_report_wf_carpet_report": ty.Any,
            "dwi_report_wf_fa_report": ty.Any,
            "dwi_report_wf_heatmap_report": ty.Any,
            "dwi_report_wf_md_report": ty.Any,
            "dwi_report_wf_noise_report": ty.Any,
            "dwi_report_wf_snr_report": ty.Any,
            "dwi_report_wf_spikes_report": ty.Any,
            "iqms_wf_noise_floor": ty.Any,
            "iqms_wf_out_file": ty.Any,
        },
        bvals=bvals,
        bvecs=bvecs,
        in_file=in_file,
        qspace_neighbors=qspace_neighbors,
    )

    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation

    workflow.add(
        SanitizeImage(
            max_32bit=exec_float32,
            n_volumes_to_discard=0,
            in_file=workflow.lzin.in_file,
            name="sanitize",
        )
    )
    # Workflow --------------------------------------------------------
    # Read metadata & bvec/bval, estimate number of shells, extract and split B0s

    workflow.add(NumberOfShells(in_bvals=workflow.lzin.bvals, name="shells"))
    workflow.add(
        ExtractOrientations(in_file=workflow.sanitize.lzout.out_file, name="get_lowb")
    )
    # Generate B0 reference
    workflow.add(
        RobustAverage(
            mc_method=None, in_file=workflow.sanitize.lzout.out_file, name="dwi_ref"
        )
    )
    workflow.add(
        Volreg(
            args="-Fourier -twopass",
            outputtype="NIFTI_GZ",
            zpad=4,
            basefile=workflow.dwi_ref.lzout.out_file,
            in_file=workflow.get_lowb.lzout.out_file,
            name="hmc_b0",
        )
    )
    # Calculate brainmask
    workflow.add(
        dmri_bmsk_workflow(
            omp_nthreads=nipype_omp_nthreads,
            in_files=workflow.dwi_ref.lzout.out_file,
            name="dmri_bmsk",
        )
    )
    # HMC: head motion correct
    workflow.add(
        hmc_workflow(
            wf_fd_radius=wf_fd_radius, in_bvec=workflow.lzin.bvecs, name="hmcwf"
        )
    )
    workflow.add(
        ExtractOrientations(
            in_bvec_file=workflow.lzin.bvecs,
            in_file=workflow.hmcwf.lzout.out_file,
            indices=workflow.shells.lzout.b_indices,
            name="get_hmc_shells",
        )
    )
    # Split shells and compute some stats
    workflow.add(
        WeightedStat(in_weights=workflow.shells.lzout.b_masks, name="averages")
    )
    workflow.add(
        WeightedStat(
            stat="std", in_weights=workflow.shells.lzout.b_masks, name="stddev"
        )
    )
    workflow.add(
        DwiDenoise(
            noise="noisemap.nii.gz",
            nthreads=nipype_omp_nthreads,
            mask=workflow.dmri_bmsk.lzout.out_mask,
            name="dwidenoise",
        )
    )
    workflow.add(
        CorrectSignalDrift(
            brainmask_file=workflow.dmri_bmsk.lzout.out_mask,
            bval_file=workflow.lzin.bvals,
            full_epi=workflow.sanitize.lzout.out_file,
            in_file=workflow.hmc_b0.lzout.out_file,
            name="drift",
        )
    )
    workflow.add(
        SpikingVoxelsMask(
            b_masks=workflow.shells.lzout.b_masks,
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            in_file=workflow.sanitize.lzout.out_file,
            name="sp_mask",
        )
    )
    # Fit DTI/DKI model
    workflow.add(
        DiffusionModel(
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            bvals=workflow.shells.lzout.out_data,
            bvec_file=workflow.lzin.bvecs,
            in_file=workflow.dwidenoise.lzout.out,
            n_shells=workflow.shells.lzout.n_shells,
            name="dwimodel",
        )
    )
    # Calculate CC mask
    workflow.add(
        CCSegmentation(
            in_cfa=workflow.dwimodel.lzout.out_cfa,
            in_fa=workflow.dwimodel.lzout.out_fa,
            name="cc_mask",
        )
    )
    # Run PIESNO noise estimation
    workflow.add(PIESNO(in_file=workflow.sanitize.lzout.out_file, name="piesno"))
    # EPI to MNI registration
    workflow.add(
        epi_mni_align(
            nipype_omp_nthreads=nipype_omp_nthreads,
            wf_species=wf_species,
            exec_ants_float=exec_ants_float,
            exec_debug=exec_debug,
            nipype_nprocs=nipype_nprocs,
            wf_template_id=wf_template_id,
            epi_mask=workflow.dmri_bmsk.lzout.out_mask,
            epi_mean=workflow.dwi_ref.lzout.out_file,
            name="spatial_norm",
        )
    )
    # Compute IQMs
    workflow.add(
        compute_iqms(
            in_noise=workflow.dwidenoise.lzout.noise,
            in_bvec=workflow.get_hmc_shells.lzout.out_bvec,
            in_shells=workflow.get_hmc_shells.lzout.out_file,
            b_values_shells=workflow.shells.lzout.b_values,
            wm_mask=workflow.cc_mask.lzout.wm_finalmask,
            cc_mask=workflow.cc_mask.lzout.out_mask,
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            in_md=workflow.dwimodel.lzout.out_md,
            in_fa_degenerate=workflow.dwimodel.lzout.out_fa_degenerate,
            in_fa_nans=workflow.dwimodel.lzout.out_fa_nans,
            in_cfa=workflow.dwimodel.lzout.out_cfa,
            in_fa=workflow.dwimodel.lzout.out_fa,
            in_bvec_diff=workflow.hmcwf.lzout.out_bvec_diff,
            in_bvec_rotated=workflow.hmcwf.lzout.out_bvec,
            framewise_displacement=workflow.hmcwf.lzout.out_fd,
            piesno_sigma=workflow.piesno.lzout.sigma,
            spikes_mask=workflow.sp_mask.lzout.out_mask,
            qspace_neighbors=workflow.lzin.qspace_neighbors,
            b_values_file=workflow.lzin.bvals,
            in_file=workflow.lzin.in_file,
            name="iqms_wf",
        )
    )
    # Generate outputs
    workflow.add(
        init_dwi_report_wf(
            exec_verbose_reports=exec_verbose_reports,
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_fd_thres=wf_fd_thres,
            exec_work_dir=exec_work_dir,
            wf_species=wf_species,
            wf_fft_spikes_detector=wf_fft_spikes_detector,
            in_parcellation=workflow.spatial_norm.lzout.epi_parc,
            in_md=workflow.dwimodel.lzout.out_md,
            in_fa=workflow.dwimodel.lzout.out_fa,
            in_epi=workflow.drift.lzout.out_full_file,
            in_stdmap=workflow.stddev.lzout.out_file,
            in_avgmap=workflow.averages.lzout.out_file,
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            in_bdict=workflow.shells.lzout.b_dict,
            name="dwi_report_wf",
        )
    )
    # fmt: off

    @pydra.mark.task
    def shells_b_masks_to_dwi_ref_t_mask_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    workflow.add(shells_b_masks_to_dwi_ref_t_mask_callable(in_=workflow.shells.lzout.b_masks, name="shells_b_masks_to_dwi_ref_t_mask_callable"))

    workflow.dwi_ref.inputs.t_mask = workflow.shells_b_masks_to_dwi_ref_t_mask_callable.lzout.out

    @pydra.mark.task
    def shells_b_indices_to_get_lowb_indices_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    workflow.add(shells_b_indices_to_get_lowb_indices_callable(in_=workflow.shells.lzout.b_indices, name="shells_b_indices_to_get_lowb_indices_callable"))

    workflow.get_lowb.inputs.indices = workflow.shells_b_indices_to_get_lowb_indices_callable.lzout.out

    @pydra.mark.task
    def shells_b_indices_to_drift_b0_ixs_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    workflow.add(shells_b_indices_to_drift_b0_ixs_callable(in_=workflow.shells.lzout.b_indices, name="shells_b_indices_to_drift_b0_ixs_callable"))

    workflow.drift.inputs.b0_ixs = workflow.shells_b_indices_to_drift_b0_ixs_callable.lzout.out
    workflow.hmcwf.inputs.in_file = workflow.drift.lzout.out_full_file
    workflow.averages.inputs.in_file = workflow.drift.lzout.out_full_file
    workflow.stddev.inputs.in_file = workflow.drift.lzout.out_full_file

    @pydra.mark.task
    def averages_out_file_to_hmcwf_reference_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    workflow.add(averages_out_file_to_hmcwf_reference_callable(in_=workflow.averages.lzout.out_file, name="averages_out_file_to_hmcwf_reference_callable"))

    workflow.hmcwf.inputs.reference = workflow.averages_out_file_to_hmcwf_reference_callable.lzout.out
    workflow.dwidenoise.inputs.dwi = workflow.drift.lzout.out_full_file

    @pydra.mark.task
    def averages_out_file_to_iqms_wf_in_b0_callable(in_: ty.Any) -> ty.Any:
        return _first(in_)

    workflow.add(averages_out_file_to_iqms_wf_in_b0_callable(in_=workflow.averages.lzout.out_file, name="averages_out_file_to_iqms_wf_in_b0_callable"))

    workflow.iqms_wf.inputs.in_b0 = workflow.averages_out_file_to_iqms_wf_in_b0_callable.lzout.out
    # fmt: on
    workflow.set_output([("iqms_wf_out_file", workflow.iqms_wf.lzout.out_file)])
    workflow.set_output([("iqms_wf_noise_floor", workflow.iqms_wf.lzout.noise_floor)])
    workflow.set_output(
        [("dwi_report_wf_spikes_report", workflow.dwi_report_wf.lzout.spikes_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_carpet_report", workflow.dwi_report_wf.lzout.carpet_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_heatmap_report", workflow.dwi_report_wf.lzout.heatmap_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_md_report", workflow.dwi_report_wf.lzout.md_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_fa_report", workflow.dwi_report_wf.lzout.fa_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_noise_report", workflow.dwi_report_wf.lzout.noise_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_bmask_report", workflow.dwi_report_wf.lzout.bmask_report)]
    )
    workflow.set_output(
        [("dwi_report_wf_snr_report", workflow.dwi_report_wf.lzout.snr_report)]
    )

    return workflow


def hmc_workflow(
    in_bvec=attrs.NOTHING,
    in_file=attrs.NOTHING,
    name="dMRI_HMC",
    reference=attrs.NOTHING,
    wf_fd_radius=50,
):
    """
    Create a :abbr:`HMC (head motion correction)` workflow for dMRI.

    .. workflow::

        from mriqc.workflows.diffusion.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        FramewiseDisplacement,
    )
    from pydra.tasks.afni.auto import Volreg
    from pydra.tasks.mriqc.interfaces.diffusion import RotateVectors

    workflow = Workflow(
        name=name,
        input_spec={"in_bvec": ty.Any, "in_file": ty.Any, "reference": ty.Any},
        output_spec={
            "out_bvec": ty.Any,
            "out_bvec_diff": ty.Any,
            "out_fd": ty.Any,
            "out_file": ty.Any,
        },
        in_bvec=in_bvec,
        in_file=in_file,
        reference=reference,
    )

    # calculate hmc parameters
    workflow.add(
        Volreg(
            args="-Fourier -twopass",
            outputtype="NIFTI_GZ",
            zpad=4,
            basefile=workflow.lzin.reference,
            in_file=workflow.lzin.in_file,
            name="hmc",
        )
    )
    workflow.add(
        RotateVectors(
            in_file=workflow.lzin.in_bvec,
            reference=workflow.lzin.reference,
            transforms=workflow.hmc.lzout.oned_matrix_save,
            name="bvec_rot",
        )
    )
    # Compute the frame-wise displacement
    workflow.add(
        FramewiseDisplacement(
            normalize=False,
            parameter_source="AFNI",
            radius=wf_fd_radius,
            in_file=workflow.hmc.lzout.oned_file,
            name="fdnode",
        )
    )
    # fmt: off
    workflow.set_output([('out_file', workflow.hmc.lzout.out_file)])
    workflow.set_output([('out_fd', workflow.fdnode.lzout.out_file)])
    workflow.set_output([('out_bvec', workflow.bvec_rot.lzout.out_bvec)])
    workflow.set_output([('out_bvec_diff', workflow.bvec_rot.lzout.out_diff)])
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

        from mriqc.workflows.diffusion.base import epi_mni_align
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


def compute_iqms(
    b_values_file=attrs.NOTHING,
    b_values_shells=attrs.NOTHING,
    brain_mask=attrs.NOTHING,
    cc_mask=attrs.NOTHING,
    framewise_displacement=attrs.NOTHING,
    in_b0=attrs.NOTHING,
    in_bvec=attrs.NOTHING,
    in_bvec_diff=attrs.NOTHING,
    in_bvec_rotated=attrs.NOTHING,
    in_cfa=attrs.NOTHING,
    in_fa=attrs.NOTHING,
    in_fa_degenerate=attrs.NOTHING,
    in_fa_nans=attrs.NOTHING,
    in_file=attrs.NOTHING,
    in_md=attrs.NOTHING,
    in_noise=attrs.NOTHING,
    in_shells=attrs.NOTHING,
    name="ComputeIQMs",
    piesno_sigma=attrs.NOTHING,
    qspace_neighbors=attrs.NOTHING,
    spikes_mask=attrs.NOTHING,
    wm_mask=attrs.NOTHING,
):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.diffusion.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.niworkflows.interfaces.bids import ReadSidecarJSON
    from pydra.tasks.mriqc.interfaces import IQMFileSink
    from pydra.tasks.mriqc.interfaces.diffusion import DiffusionQC
    from pydra.tasks.mriqc.interfaces.reports import AddProvenance

    # from mriqc.workflows.utils import _tofloat, get_fwhmx
    workflow = Workflow(
        name=name,
        input_spec={
            "b_values_file": ty.Any,
            "b_values_shells": ty.Any,
            "brain_mask": ty.Any,
            "cc_mask": ty.Any,
            "framewise_displacement": ty.Any,
            "in_b0": ty.Any,
            "in_bvec": ty.Any,
            "in_bvec_diff": ty.Any,
            "in_bvec_rotated": ty.Any,
            "in_cfa": ty.Any,
            "in_fa": ty.Any,
            "in_fa_degenerate": ty.Any,
            "in_fa_nans": ty.Any,
            "in_file": ty.Any,
            "in_md": ty.Any,
            "in_noise": ty.Any,
            "in_shells": ty.Any,
            "piesno_sigma": ty.Any,
            "qspace_neighbors": ty.Any,
            "spikes_mask": ty.Any,
            "wm_mask": ty.Any,
        },
        output_spec={"noise_floor": ty.Any, "out_file": ty.Any},
        b_values_file=b_values_file,
        b_values_shells=b_values_shells,
        brain_mask=brain_mask,
        cc_mask=cc_mask,
        framewise_displacement=framewise_displacement,
        in_b0=in_b0,
        in_bvec=in_bvec,
        in_bvec_diff=in_bvec_diff,
        in_bvec_rotated=in_bvec_rotated,
        in_cfa=in_cfa,
        in_fa=in_fa,
        in_fa_degenerate=in_fa_degenerate,
        in_fa_nans=in_fa_nans,
        in_file=in_file,
        in_md=in_md,
        in_noise=in_noise,
        in_shells=in_shells,
        piesno_sigma=piesno_sigma,
        qspace_neighbors=qspace_neighbors,
        spikes_mask=spikes_mask,
        wm_mask=wm_mask,
    )

    workflow.add(
        FunctionTask(
            func=_estimate_sigma,
            in_file=workflow.lzin.in_noise,
            mask=workflow.lzin.brain_mask,
            name="estimate_sigma",
        )
    )

    workflow.add(
        DiffusionQC(
            brain_mask=workflow.lzin.brain_mask,
            cc_mask=workflow.lzin.cc_mask,
            in_b0=workflow.lzin.in_b0,
            in_bval_file=workflow.lzin.b_values_file,
            in_bvec=workflow.lzin.in_bvec,
            in_bvec_diff=workflow.lzin.in_bvec_diff,
            in_bvec_rotated=workflow.lzin.in_bvec_rotated,
            in_cfa=workflow.lzin.in_cfa,
            in_fa=workflow.lzin.in_fa,
            in_fa_degenerate=workflow.lzin.in_fa_degenerate,
            in_fa_nans=workflow.lzin.in_fa_nans,
            in_fd=workflow.lzin.framewise_displacement,
            in_file=workflow.lzin.in_file,
            in_md=workflow.lzin.in_md,
            in_shells=workflow.lzin.in_shells,
            in_shells_bval=workflow.lzin.b_values_shells,
            piesno_sigma=workflow.lzin.piesno_sigma,
            qspace_neighbors=workflow.lzin.qspace_neighbors,
            spikes_mask=workflow.lzin.spikes_mask,
            wm_mask=workflow.lzin.wm_mask,
            name="measures",
        )
    )

    # Save to JSON file

    # fmt: off




    workflow.set_output([('out_file', workflow.measures.lzout.out_qc)])
    workflow.set_output([('noise_floor', workflow.estimate_sigma.lzout.out)])
    # fmt: on

    return workflow


def _bvals_report(in_file):

    import numpy as np

    bvals = [
        round(float(val), 2) for val in np.unique(np.round(np.loadtxt(in_file), 2))
    ]
    if len(bvals) > 10:
        return "Likely DSI"
    return bvals


def _estimate_sigma(in_file, mask):

    import nibabel as nb
    import numpy as np

    msk = nb.load(mask).get_fdata() > 0.5
    return round(
        float(np.median(nb.load(in_file).get_fdata()[msk])),
        6,
    )


def _filter_metadata(
    in_dict,
    keys=(
        "global",
        "dcmmeta_affine",
        "dcmmeta_reorient_transform",
        "dcmmeta_shape",
        "dcmmeta_slice_dim",
        "dcmmeta_version",
        "time",
    ),
):
    """Drop large and partially redundant objects generated by dcm2niix."""
    for key in keys:
        in_dict.pop(key, None)
    return in_dict


def _first(inlist):

    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist
