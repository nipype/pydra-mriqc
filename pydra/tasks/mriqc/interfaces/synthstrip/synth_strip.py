from fileformats.generic import File
import logging
from pathlib import Path
from pydra.engine import ShellCommandTask, specs


logger = logging.getLogger(__name__)


input_fields = [
    (
        "in_file",
        File,
        {
            "help_string": "Input image to be brain extracted",
            "argstr": "-i {in_file}",
            "mandatory": True,
            "copyfile": True,
        },
    ),
    ("use_gpu", bool, False, {"help_string": "Use GPU", "argstr": "-g"}),
    (
        "model",
        File,
        "/Applications/freesurfer/7.4.1/models/synthstrip.1.pt",
        {"help_string": "file containing model's weights", "argstr": "--model {model}"},
    ),
    (
        "border_mm",
        int,
        1,
        {"help_string": "Mask border threshold in mm", "argstr": "-b {border_mm}"},
    ),
    (
        "out_file",
        Path,
        {
            "help_string": "store brain-extracted input to file",
            "argstr": "-o {out_file}",
            "output_file_template": "{in_file}_desc-brain.nii.gz",
        },
    ),
    (
        "out_mask",
        Path,
        {
            "help_string": "store brainmask to file",
            "argstr": "-m {out_mask}",
            "output_file_template": "{in_file}_desc-brain_mask.nii.gz",
        },
    ),
    (
        "num_threads",
        int,
        {"help_string": "Number of threads", "argstr": "-n {num_threads}"},
    ),
]
SynthStrip_input_spec = specs.SpecInfo(
    name="Input", fields=input_fields, bases=(specs.ShellSpec,)
)

output_fields = []
SynthStrip_output_spec = specs.SpecInfo(
    name="Output", fields=output_fields, bases=(specs.ShellOutSpec,)
)


class SynthStrip(ShellCommandTask):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.synthstrip.synth_strip import SynthStrip

    """

    input_spec = SynthStrip_input_spec
    output_spec = SynthStrip_output_spec
    executable = "synthstrip"
