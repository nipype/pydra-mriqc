import attrs
from fileformats.medimage import NiftiGzX, T1Weighted
import logging
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, SpecInfo
from pydra.engine.task import FunctionTask
import pydra.mark
from pydra.tasks.mriqc.interfaces import (
    ArtifactMask,
    ComputeQI2,
    ConformImage,
    RotationMask,
    StructuralQC,
)
from pydra.tasks.mriqc.workflows.anatomical.output import init_anat_report_wf
from pydra.tasks.mriqc.workflows.utils import get_fwhmx
from pydra.tasks.niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from templateflow.api import get as get_template
import typing as ty


logger = logging.getLogger(__name__)


def anat_qc_workflow(
    exec_ants_float=False,
    exec_datalad_get=True,
    exec_debug=False,
    exec_no_sub=False,
    exec_verbose_reports=False,
    exec_work_dir=None,
    in_file=attrs.NOTHING,
    modality=attrs.NOTHING,
    name="anatMRIQC",
    nipype_omp_nthreads=12,
    wf_inputs=None,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images

    .. workflow::

        import os.path as op
        from mriqc.workflows.anatomical.base import anat_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = anat_qc_workflow()

    """
    from pydra.tasks.mriqc.workflows.shared import synthstrip_wf

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    # Initialize workflow
    workflow = Workflow(
        name=name,
        input_spec={"in_file": NiftiGzX[T1Weighted], "modality": str},
        output_spec={
            "anat_report_wf_airmask_report": ty.Any,
            "anat_report_wf_artmask_report": ty.Any,
            "anat_report_wf_bg_report": ty.Any,
            "anat_report_wf_bmask_report": ty.Any,
            "anat_report_wf_headmask_report": ty.Any,
            "anat_report_wf_segm_report": ty.Any,
            "anat_report_wf_zoom_report": ty.Any,
            "iqmswf_noise_report": ty.Any,
            "norm_report": ty.Any,
        },
        in_file=in_file,
        modality=modality,
    )

    # Define workflow, inputs and outputs
    # 0. Get data

    # 1. Reorient anatomical image
    workflow.add(
        ConformImage(check_dtype=False, in_file=workflow.lzin.in_file, name="to_ras")
    )
    # 2. species specific skull-stripping
    if wf_species.lower() == "human":
        workflow.add(
            synthstrip_wf(
                omp_nthreads=nipype_omp_nthreads,
                in_files=workflow.to_ras.lzout.out_file,
                name="skull_stripping",
            )
        )
        ss_bias_field = "outputnode.bias_image"
    else:
        from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf

        skull_stripping = init_rodent_brain_extraction_wf(template_id=wf_template_id)
        ss_bias_field = "final_n4.bias_image"
    # 3. Head mask
    workflow.add(
        headmsk_wf(omp_nthreads=nipype_omp_nthreads, wf_species=wf_species, name="hmsk")
    )
    # 4. Spatial Normalization, using ANTs
    workflow.add(
        spatial_normalization(
            nipype_omp_nthreads=nipype_omp_nthreads,
            exec_debug=exec_debug,
            wf_species=wf_species,
            wf_template_id=wf_template_id,
            exec_ants_float=exec_ants_float,
            modality=workflow.lzin.modality,
            name="spatial_norm",
        )
    )
    # 5. Air mask (with and without artifacts)
    workflow.add(
        airmsk_wf(
            ind2std_xfm=workflow.spatial_norm.lzout.ind2std_xfm,
            in_file=workflow.to_ras.lzout.out_file,
            head_mask=workflow.hmsk.lzout.out_file,
            name="amw",
        )
    )
    # 6. Brain tissue segmentation
    workflow.add(
        init_brain_tissue_segmentation(
            nipype_omp_nthreads=nipype_omp_nthreads,
            std_tpms=workflow.spatial_norm.lzout.out_tpms,
            in_file=workflow.hmsk.lzout.out_denoised,
            name="bts",
        )
    )
    # 7. Compute IQMs
    workflow.add(
        compute_iqms(
            wf_species=wf_species,
            std_tpms=workflow.spatial_norm.lzout.out_tpms,
            in_ras=workflow.to_ras.lzout.out_file,
            airmask=workflow.amw.lzout.air_mask,
            hatmask=workflow.amw.lzout.hat_mask,
            artmask=workflow.amw.lzout.art_mask,
            rotmask=workflow.amw.lzout.rot_mask,
            segmentation=workflow.bts.lzout.out_segm,
            pvms=workflow.bts.lzout.out_pvms,
            headmask=workflow.hmsk.lzout.out_file,
            name="iqmswf",
        )
    )
    # Reports
    workflow.add(
        init_anat_report_wf(
            exec_verbose_reports=exec_verbose_reports,
            wf_species=wf_species,
            exec_work_dir=exec_work_dir,
            in_ras=workflow.to_ras.lzout.out_file,
            headmask=workflow.hmsk.lzout.out_file,
            airmask=workflow.amw.lzout.air_mask,
            artmask=workflow.amw.lzout.art_mask,
            segmentation=workflow.bts.lzout.out_segm,
            name="anat_report_wf",
        )
    )
    # Connect all nodes
    # fmt: off

    workflow.hmsk.inputs.in_file = workflow.skull_stripping.lzout.out_corrected
    workflow.hmsk.inputs.brainmask = workflow.skull_stripping.lzout.out_mask
    workflow.bts.inputs.brainmask = workflow.skull_stripping.lzout.out_mask
    workflow.spatial_norm.inputs.moving_image = workflow.skull_stripping.lzout.out_corrected
    workflow.spatial_norm.inputs.moving_mask = workflow.skull_stripping.lzout.out_mask
    workflow.hmsk.inputs.in_tpms = workflow.spatial_norm.lzout.out_tpms

    workflow.iqmswf.inputs.inu_corrected = workflow.skull_stripping.lzout.out_corrected
    workflow.iqmswf.inputs.in_inu = workflow.skull_stripping.lzout.bias_image
    workflow.iqmswf.inputs.brainmask = workflow.skull_stripping.lzout.out_mask

    workflow.anat_report_wf.inputs.brainmask = workflow.skull_stripping.lzout.out_mask

    # fmt: on
    # Upload metrics
    if not exec_no_sub:
        from pydra.tasks.mriqc.interfaces.webapi import UploadIQMs

        pass
        # fmt: off
        pass
        pass
        # fmt: on
    workflow.set_output([("norm_report", workflow.spatial_norm.lzout.report)])
    workflow.set_output([("iqmswf_noise_report", workflow.iqmswf.lzout.noise_report)])
    workflow.set_output(
        [("anat_report_wf_bmask_report", workflow.anat_report_wf.lzout.bmask_report)]
    )
    workflow.set_output(
        [("anat_report_wf_segm_report", workflow.anat_report_wf.lzout.segm_report)]
    )
    workflow.set_output(
        [
            (
                "anat_report_wf_artmask_report",
                workflow.anat_report_wf.lzout.artmask_report,
            )
        ]
    )
    workflow.set_output(
        [("anat_report_wf_bg_report", workflow.anat_report_wf.lzout.bg_report)]
    )
    workflow.set_output(
        [("anat_report_wf_zoom_report", workflow.anat_report_wf.lzout.zoom_report)]
    )
    workflow.set_output(
        [
            (
                "anat_report_wf_headmask_report",
                workflow.anat_report_wf.lzout.headmask_report,
            )
        ]
    )
    workflow.set_output(
        [
            (
                "anat_report_wf_airmask_report",
                workflow.anat_report_wf.lzout.airmask_report,
            )
        ]
    )

    return workflow


def airmsk_wf(
    head_mask=attrs.NOTHING,
    in_file=attrs.NOTHING,
    ind2std_xfm=attrs.NOTHING,
    name="AirMaskWorkflow",
):
    """
    Calculate air, artifacts and "hat" masks to evaluate noise in the background.

    This workflow mostly addresses the implementation of Step 1 in [Mortamet2009]_.
    This work proposes to look at the signal distribution in the background, where
    no signals are expected, to evaluate the spread of the noise.
    It is in the background where [Mortamet2009]_ proposed to also look at the presence
    of ghosts and artifacts, where they are very easy to isolate.

    However, [Mortamet2009]_ proposes not to look at the background around the face
    because of the likely signal leakage through the phase-encoding axis sourcing from
    eyeballs (and their motion).
    To avoid that, [Mortamet2009]_ proposed atlas-based identification of two landmarks
    (nasion and cerebellar projection on to the occipital bone).
    MRIQC, for simplicity, used a such a mask created in MNI152NLin2009cAsym space and
    projected it on to the individual.
    Such a solution is inadequate because it doesn't drop full in-plane slices as there
    will be a large rotation of the individual's tilt of the head with respect to the
    template.
    The new implementation (23.1.x series) follows [Mortamet2009]_ more closely,
    projecting the two landmarks from the template space and leveraging
    *NiTransforms* to do that.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical.base import airmsk_wf
        with mock_config():
            wf = airmsk_wf()

    """
    workflow = Workflow(
        name=name,
        input_spec={"head_mask": ty.Any, "in_file": ty.Any, "ind2std_xfm": ty.Any},
        output_spec={
            "air_mask": ty.Any,
            "art_mask": ty.Any,
            "hat_mask": ty.Any,
            "rot_mask": ty.Any,
        },
        head_mask=head_mask,
        in_file=in_file,
        ind2std_xfm=ind2std_xfm,
    )

    workflow.add(RotationMask(in_file=workflow.lzin.in_file, name="rotmsk"))
    workflow.add(
        ArtifactMask(
            head_mask=workflow.lzin.head_mask,
            in_file=workflow.lzin.in_file,
            ind2std_xfm=workflow.lzin.ind2std_xfm,
            name="qi1",
        )
    )
    # fmt: off
    workflow.set_output([('hat_mask', workflow.qi1.lzout.out_hat_msk)])
    workflow.set_output([('air_mask', workflow.qi1.lzout.out_air_msk)])
    workflow.set_output([('art_mask', workflow.qi1.lzout.out_art_msk)])
    workflow.set_output([('rot_mask', workflow.rotmsk.lzout.out_file)])
    # fmt: on

    return workflow


def headmsk_wf(
    brainmask=attrs.NOTHING,
    in_file=attrs.NOTHING,
    in_tpms=attrs.NOTHING,
    name="HeadMaskWorkflow",
    omp_nthreads=1,
    wf_species="human",
):
    """
    Computes a head mask as in [Mortamet2009]_.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical.base import headmsk_wf
        with mock_config():
            wf = headmsk_wf()

    """
    from pydra.tasks.niworkflows.interfaces.nibabel import ApplyMask

    workflow = Workflow(
        name=name,
        input_spec={"brainmask": ty.Any, "in_file": ty.Any, "in_tpms": ty.Any},
        output_spec={"out_denoised": ty.Any, "out_file": ty.Any},
        brainmask=brainmask,
        in_file=in_file,
        in_tpms=in_tpms,
    )

    def _select_wm(inlist):
        return [f for f in inlist if "WM" in f][0]

    workflow.add(
        FunctionTask(
            func=_enhance,
            input_spec=SpecInfo(
                name="FunctionIn",
                bases=(BaseSpec,),
                fields=[("in_file", ty.Any), ("wm_tpm", ty.Any)],
            ),
            output_spec=SpecInfo(
                name="FunctionOut", bases=(BaseSpec,), fields=[("out_file", ty.Any)]
            ),
            in_file=workflow.lzin.in_file,
            wm_tpm=workflow.lzin.in_tpms,
            name="enhance",
        )
    )
    workflow.add(
        FunctionTask(
            func=image_gradient,
            input_spec=SpecInfo(
                name="FunctionIn",
                bases=(BaseSpec,),
                fields=[("in_file", ty.Any), ("brainmask", ty.Any), ("sigma", ty.Any)],
            ),
            output_spec=SpecInfo(
                name="FunctionOut", bases=(BaseSpec,), fields=[("out_file", ty.Any)]
            ),
            brainmask=workflow.lzin.brainmask,
            in_file=workflow.enhance.lzout.out_file,
            name="gradient",
        )
    )
    workflow.add(
        FunctionTask(
            func=gradient_threshold,
            input_spec=SpecInfo(
                name="FunctionIn",
                bases=(BaseSpec,),
                fields=[
                    ("in_file", ty.Any),
                    ("brainmask", ty.Any),
                    ("aniso", ty.Any),
                    ("thresh", ty.Any),
                ],
            ),
            output_spec=SpecInfo(
                name="FunctionOut", bases=(BaseSpec,), fields=[("out_file", ty.Any)]
            ),
            brainmask=workflow.lzin.brainmask,
            in_file=workflow.gradient.lzout.out_file,
            name="thresh",
        )
    )
    if wf_species != "human":
        workflow.gradient.inputs.sigma = 3.0
        workflow.thresh.inputs.aniso = True
        workflow.thresh.inputs.thresh = 4.0
    workflow.add(
        ApplyMask(
            in_file=workflow.enhance.lzout.out_file,
            in_mask=workflow.lzin.brainmask,
            name="apply_mask",
        )
    )
    # fmt: off
    workflow.enhance.inputs.wm_tpm = workflow.lzin.in_tpms
    workflow.set_output([('out_file', workflow.thresh.lzout.out_file)])
    workflow.set_output([('out_denoised', workflow.apply_mask.lzout.out_file)])
    # fmt: on

    return workflow


def init_brain_tissue_segmentation(
    brainmask=attrs.NOTHING,
    in_file=attrs.NOTHING,
    name="brain_tissue_segmentation",
    nipype_omp_nthreads=12,
    std_tpms=attrs.NOTHING,
):
    """
    Setup a workflow for brain tissue segmentation.

    .. workflow::

        from mriqc.workflows.anatomical.base import init_brain_tissue_segmentation
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_brain_tissue_segmentation()

    """
    from pydra.tasks.ants.auto import Atropos

    def _format_tpm_names(in_files, fname_string=None):
        import glob
        from pathlib import Path
        import nibabel as nb

        out_path = Path.cwd().absolute()
        # copy files to cwd and rename iteratively
        for count, fname in enumerate(in_files):
            img = nb.load(fname)
            extension = "".join(Path(fname).suffixes)
            out_fname = f"priors_{1 + count:02}{extension}"
            nb.save(img, Path(out_path, out_fname))
        if fname_string is None:
            fname_string = f"priors_%02d{extension}"
        out_files = [
            str(prior)
            for prior in glob.glob(str(Path(out_path, f"priors*{extension}")))
        ]
        # return path with c-style format string for Atropos
        file_format = str(Path(out_path, fname_string))
        return file_format, out_files

    workflow = Workflow(
        name=name,
        input_spec={"brainmask": ty.Any, "in_file": ty.Any, "std_tpms": ty.Any},
        output_spec={"out_pvms": ty.Any, "out_segm": ty.Any},
        brainmask=brainmask,
        in_file=in_file,
        std_tpms=std_tpms,
    )

    workflow.add(
        FunctionTask(
            execution={"keep_inputs": True, "remove_unnecessary_outputs": False},
            func=_format_tpm_names,
            input_spec=SpecInfo(
                name="FunctionIn", bases=(BaseSpec,), fields=[("in_files", ty.Any)]
            ),
            output_spec=SpecInfo(
                name="FunctionOut", bases=(BaseSpec,), fields=[("file_format", ty.Any)]
            ),
            in_files=workflow.lzin.std_tpms,
            name="format_tpm_names",
        )
    )
    workflow.add(
        Atropos(
            initialization="PriorProbabilityImages",
            mrf_radius=[1, 1, 1],
            mrf_smoothing_factor=0.01,
            num_threads=nipype_omp_nthreads,
            number_of_tissue_classes=3,
            out_classified_image_name="segment.nii.gz",
            output_posteriors_name_template="segment_%02d.nii.gz",
            prior_weighting=0.1,
            save_posteriors=True,
            intensity_images=workflow.lzin.in_file,
            mask_image=workflow.lzin.brainmask,
            name="segment",
        )
    )
    # fmt: off

    @pydra.mark.task
    def format_tpm_names_file_format_to_segment_prior_image_callable(in_: ty.Any) -> ty.Any:
        return _pop(in_)

    workflow.add(format_tpm_names_file_format_to_segment_prior_image_callable(in_=workflow.format_tpm_names.lzout.file_format, name="format_tpm_names_file_format_to_segment_prior_image_callable"))

    workflow.segment.inputs.prior_image = workflow.format_tpm_names_file_format_to_segment_prior_image_callable.lzout.out
    workflow.set_output([('out_segm', workflow.segment.lzout.classified_image)])
    workflow.set_output([('out_pvms', workflow.segment.lzout.posteriors)])
    # fmt: on

    return workflow


def spatial_normalization(
    exec_ants_float=False,
    exec_debug=False,
    modality=attrs.NOTHING,
    moving_image=attrs.NOTHING,
    moving_mask=attrs.NOTHING,
    name="SpatialNormalization",
    nipype_omp_nthreads=12,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """Create a simplified workflow to perform fast spatial normalization."""
    from pydra.tasks.niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )

    # Have the template id handy
    tpl_id = wf_template_id
    # Define workflow interface
    workflow = Workflow(
        name=name,
        input_spec={"modality": ty.Any, "moving_image": ty.Any, "moving_mask": ty.Any},
        output_spec={"ind2std_xfm": ty.Any, "out_tpms": ty.Any, "report": ty.Any},
        modality=modality,
        moving_image=moving_image,
        moving_mask=moving_mask,
    )

    # Spatial normalization
    workflow.add(
        RobustMNINormalization(
            flavor=["testing", "fast"][exec_debug],
            float=exec_ants_float,
            generate_report=True,
            num_threads=nipype_omp_nthreads,
            template=tpl_id,
            moving_image=workflow.lzin.moving_image,
            moving_mask=workflow.lzin.moving_mask,
            reference=workflow.lzin.modality,
            name="norm",
        )
    )
    if wf_species.lower() == "human":
        workflow.norm.inputs.reference_mask = str(
            get_template(tpl_id, resolution=2, desc="brain", suffix="mask")
        )
    else:
        workflow.norm.inputs.reference_image = str(get_template(tpl_id, suffix="T2w"))
        workflow.norm.inputs.reference_mask = str(
            get_template(tpl_id, desc="brain", suffix="mask")[0]
        )
    # Project standard TPMs into T1w space
    workflow.add(
        ApplyTransforms(
            default_value=0,
            dimension=3,
            float=exec_ants_float,
            interpolation="Gaussian",
            reference_image=workflow.lzin.moving_image,
            transforms=workflow.norm.lzout.inverse_composite_transform,
            name="tpms_std2t1w",
        )
    )
    workflow.tpms_std2t1w.inputs.input_image = [
        str(p)
        for p in get_template(
            wf_template_id,
            suffix="probseg",
            resolution=(1 if wf_species.lower() == "human" else None),
            label=["CSF", "GM", "WM"],
        )
    ]
    # fmt: off
    workflow.set_output([('ind2std_xfm', workflow.norm.lzout.composite_transform)])
    workflow.set_output([('report', workflow.norm.lzout.out_report)])
    workflow.set_output([('out_tpms', workflow.tpms_std2t1w.lzout.output_image)])
    # fmt: on

    return workflow


def compute_iqms(
    airmask=attrs.NOTHING,
    artmask=attrs.NOTHING,
    brainmask=attrs.NOTHING,
    hatmask=attrs.NOTHING,
    headmask=attrs.NOTHING,
    in_inu=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    inu_corrected=attrs.NOTHING,
    name="ComputeIQMs",
    pvms=attrs.NOTHING,
    rotmask=attrs.NOTHING,
    segmentation=attrs.NOTHING,
    std_tpms=attrs.NOTHING,
    wf_species="human",
):
    """
    Setup the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.anatomical.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.niworkflows.interfaces.bids import ReadSidecarJSON
    from pydra.tasks.mriqc.interfaces.anatomical import Harmonize
    from pydra.tasks.mriqc.workflows.utils import _tofloat

    workflow = Workflow(
        name=name,
        input_spec={
            "airmask": ty.Any,
            "artmask": ty.Any,
            "brainmask": ty.Any,
            "hatmask": ty.Any,
            "headmask": ty.Any,
            "in_inu": ty.Any,
            "in_ras": ty.Any,
            "inu_corrected": ty.Any,
            "pvms": ty.Any,
            "rotmask": ty.Any,
            "segmentation": ty.Any,
            "std_tpms": ty.Any,
        },
        output_spec={"measures": ty.Any, "noise_report": ty.Any},
        airmask=airmask,
        artmask=artmask,
        brainmask=brainmask,
        hatmask=hatmask,
        headmask=headmask,
        in_inu=in_inu,
        in_ras=in_ras,
        inu_corrected=inu_corrected,
        pvms=pvms,
        rotmask=rotmask,
        segmentation=segmentation,
        std_tpms=std_tpms,
    )

    # Extract metadata

    # Add provenance

    # AFNI check smoothing
    fwhm_interface = get_fwhmx()
    fwhm = fwhm_interface
    fwhm.name = "fwhm"
    fwhm.inputs.in_file = workflow.lzin.in_ras
    fwhm.inputs.mask = workflow.lzin.brainmask
    workflow.add(fwhm)
    # Harmonize
    workflow.add(
        Harmonize(
            in_file=workflow.lzin.inu_corrected,
            wm_mask=workflow.lzin.pvms,
            name="homog",
        )
    )
    if wf_species.lower() != "human":
        workflow.homog.inputs.erodemsk = False
        workflow.homog.inputs.thresh = 0.8
    # Mortamet's QI2
    workflow.add(
        ComputeQI2(
            air_msk=workflow.lzin.hatmask, in_file=workflow.lzin.in_ras, name="getqi2"
        )
    )
    # Compute python-coded measures
    workflow.add(
        StructuralQC(
            human=wf_species.lower() == "human",
            air_msk=workflow.lzin.airmask,
            artifact_msk=workflow.lzin.artmask,
            head_msk=workflow.lzin.headmask,
            in_bias=workflow.lzin.in_inu,
            in_file=workflow.lzin.in_ras,
            in_noinu=workflow.homog.lzout.out_file,
            in_pvms=workflow.lzin.pvms,
            in_segm=workflow.lzin.segmentation,
            mni_tpms=workflow.lzin.std_tpms,
            rot_msk=workflow.lzin.rotmask,
            name="measures",
        )
    )

    def _getwm(inlist):
        return inlist[-1]

    # fmt: off


    workflow.homog.inputs.wm_mask = workflow.lzin.pvms

    @pydra.mark.task
    def fwhm_fwhm_to_measures_in_fwhm_callable(in_: ty.Any) -> ty.Any:
        return _tofloat(in_)

    workflow.add(fwhm_fwhm_to_measures_in_fwhm_callable(in_=workflow.fwhm.lzout.fwhm, name="fwhm_fwhm_to_measures_in_fwhm_callable"))

    workflow.measures.inputs.in_fwhm = workflow.fwhm_fwhm_to_measures_in_fwhm_callable.lzout.out
    workflow.set_output([('measures', workflow.measures.lzout.out_qc)])
    workflow.set_output([('noise_report', workflow.getqi2.lzout.out_file)])

    # fmt: on

    return workflow


def _enhance(in_file, wm_tpm, out_file=None):

    import nibabel as nb
    import numpy as np
    from pydra.tasks.mriqc.workflows.utils import generate_filename

    imnii = nb.load(in_file)
    data = imnii.get_fdata(dtype=np.float32)
    range_max = np.percentile(data[data > 0], 99.98)
    excess = data > range_max
    wm_prob = nb.load(wm_tpm).get_fdata()
    wm_prob[wm_prob < 0] = 0  # Ensure no negative values
    wm_prob[excess] = 0  # Ensure no outliers are considered
    # Calculate weighted mean and standard deviation
    wm_mu = np.average(data, weights=wm_prob)
    wm_sigma = np.sqrt(np.average((data - wm_mu) ** 2, weights=wm_prob))
    # Resample signal excess pixels
    data[excess] = np.random.normal(loc=wm_mu, scale=wm_sigma, size=excess.sum())
    out_file = out_file or str(generate_filename(in_file, suffix="enhanced").absolute())
    nb.Nifti1Image(data, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def _get_mod(in_file):

    from pathlib import Path

    in_file = Path(in_file)
    extension = "".join(in_file.suffixes)
    return in_file.name.replace(extension, "").split("_")[-1]


def _pop(inlist):

    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def gradient_threshold(in_file, brainmask, thresh=15.0, out_file=None, aniso=False):
    """Compute a threshold from the histogram of the magnitude gradient image"""
    import nibabel as nb
    import numpy as np
    from scipy import ndimage as sim
    from pydra.tasks.mriqc.workflows.utils import generate_filename

    if not aniso:
        struct = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)
    else:
        # Generate an anisotropic binary structure, taking into account slice thickness
        img = nb.load(in_file)
        zooms = img.header.get_zooms()
        dist = max(zooms)
        dim = img.header["dim"][0]
        x = np.ones((5) * np.ones(dim, dtype=np.int8))
        np.put(x, x.size // 2, 0)
        dist_matrix = np.round(sim.distance_transform_edt(x, sampling=zooms), 5)
        struct = dist_matrix <= dist
    imnii = nb.load(in_file)
    hdr = imnii.header.copy()
    hdr.set_data_dtype(np.uint8)
    data = imnii.get_fdata(dtype=np.float32)
    mask = np.zeros_like(data, dtype=np.uint8)
    mask[data > thresh] = 1
    mask = sim.binary_closing(mask, struct, iterations=2).astype(np.uint8)
    mask = sim.binary_erosion(mask, sim.generate_binary_structure(3, 2)).astype(
        np.uint8
    )
    segdata = np.asanyarray(nb.load(brainmask).dataobj) > 0
    segdata = sim.binary_dilation(segdata, struct, iterations=2, border_value=1).astype(
        np.uint8
    )
    mask[segdata] = 1
    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    artmsk = np.zeros_like(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = sorted(zip(sizes, list(range(nb_labels + 1))), reverse=True)
        for _, label in ordered[2:]:
            mask[label_im == label] = 0
            artmsk[label_im == label] = 1
    mask = sim.binary_fill_holes(mask, struct).astype(
        np.uint8
    )  # pylint: disable=no-member
    out_file = out_file or str(generate_filename(in_file, suffix="gradmask").absolute())
    nb.Nifti1Image(mask, imnii.affine, hdr).to_filename(out_file)
    return out_file


def image_gradient(in_file, brainmask, sigma=4.0, out_file=None):
    """Computes the magnitude gradient of an image using numpy"""
    import nibabel as nb
    import numpy as np
    from scipy.ndimage import gaussian_gradient_magnitude as gradient
    from pydra.tasks.mriqc.workflows.utils import generate_filename

    imnii = nb.load(in_file)
    mask = np.bool_(nb.load(brainmask).dataobj)
    data = imnii.get_fdata(dtype=np.float32)
    datamax = np.percentile(data.reshape(-1), 99.5)
    data *= 100 / datamax
    data[mask] = 100
    zooms = np.array(imnii.header.get_zooms()[:3])
    sigma_xyz = 2 - zooms / min(zooms)
    grad = gradient(data, sigma * sigma_xyz)
    gradmax = np.percentile(grad.reshape(-1), 99.5)
    grad *= 100.0
    grad /= gradmax
    grad[mask] = 100
    out_file = out_file or str(generate_filename(in_file, suffix="grad").absolute())
    nb.Nifti1Image(grad, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def _binarize(in_file, threshold=0.5, out_file=None):

    import os.path as op
    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_bin{ext}")
    nii = nb.load(in_file)
    data = nii.get_fdata() > threshold
    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data.astype(np.uint8), nii.affine, hdr).to_filename(out_file)
    return out_file
