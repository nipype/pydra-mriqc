import attrs
import logging
from pydra.engine import Workflow
import typing as ty


logger = logging.getLogger(__name__)


def synthstrip_wf(in_files=attrs.NOTHING, name="synthstrip_wf", omp_nthreads=None):
    """Create a brain-extraction workflow using SynthStrip."""
    from pydra.tasks.ants.auto import N4BiasFieldCorrection
    from pydra.tasks.niworkflows.interfaces.nibabel import ApplyMask, IntensityClip
    from pydra.tasks.mriqc.interfaces.synthstrip import SynthStrip

    # truncate target intensity for N4 correction
    workflow = Workflow(
        name=name,
        input_spec={"in_files": ty.Any},
        output_spec={
            "bias_image": ty.Any,
            "out_brain": ty.Any,
            "out_corrected": ty.Any,
            "out_mask": ty.Any,
        },
        in_files=in_files,
    )

    workflow.add(
        IntensityClip(
            p_max=99.9, p_min=10, in_file=workflow.lzin.in_files, name="pre_clip"
        )
    )
    workflow.add(
        N4BiasFieldCorrection(
            copy_header=True,
            dimension=3,
            num_threads=omp_nthreads,
            rescale_intensities=True,
            input_image=workflow.pre_clip.lzout.out_file,
            name="pre_n4",
        )
    )
    workflow.add(
        N4BiasFieldCorrection(
            copy_header=True,
            dimension=3,
            n_iterations=[50] * 4,
            num_threads=omp_nthreads,
            bias_image=True,
            input_image=workflow.pre_clip.lzout.out_file,
            name="post_n4",
        )
    )
    workflow.add(
        SynthStrip(
            num_threads=omp_nthreads,
            in_file=workflow.pre_n4.lzout.output_image,
            name="synthstrip",
        )
    )
    workflow.add(
        ApplyMask(
            in_file=workflow.post_n4.lzout.output_image,
            in_mask=workflow.synthstrip.lzout.out_mask,
            name="final_masked",
        )
    )
    # fmt: off
    workflow.post_n4.inputs.weight_image = workflow.synthstrip.lzout.out_mask
    workflow.set_output([('out_brain', workflow.final_masked.lzout.out_file)])
    workflow.set_output([('bias_image', workflow.post_n4.lzout.bias_image)])
    workflow.set_output([('out_mask', workflow.synthstrip.lzout.out_mask)])
    workflow.set_output([('out_corrected', workflow.post_n4.lzout.output_image)])
    # fmt: on

    return workflow
