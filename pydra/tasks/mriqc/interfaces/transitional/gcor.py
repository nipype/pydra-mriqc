import attrs
from fileformats.generic import File
from fileformats.medimage import Nifti1
import logging
from pydra.engine import ShellCommandTask, specs


logger = logging.getLogger(__name__)


def _list_outputs(inputs=None, stdout=None, stderr=None, output_dir=None):
    inputs = attrs.asdict(inputs)
    return {"out": parsed_inputs["_gcor"]}


def out_callable(output_dir, inputs, stdout, stderr):
    outputs = _list_outputs(
        output_dir=output_dir, inputs=inputs, stdout=stdout, stderr=stderr
    )
    return outputs.get("out", attrs.NOTHING)


input_fields = [
    (
        "in_file",
        Nifti1,
        {
            "help_string": "input dataset to compute the GCOR over",
            "argstr": "-input {in_file}",
            "copyfile": False,
            "mandatory": True,
            "position": -1,
        },
    ),
    (
        "mask",
        File,
        {
            "help_string": "mask dataset, for restricting the computation",
            "argstr": "-mask {mask}",
            "copyfile": False,
        },
    ),
    (
        "nfirst",
        int,
        {
            "help_string": "specify number of initial TRs to ignore",
            "argstr": "-nfirst {nfirst}",
        },
    ),
    (
        "no_demean",
        bool,
        {
            "help_string": "do not (need to) demean as first step",
            "argstr": "-no_demean",
        },
    ),
]
GCOR_input_spec = specs.SpecInfo(
    name="Input", fields=input_fields, bases=(specs.ShellSpec,)
)

output_fields = [
    (
        "out",
        float,
        {"help_string": "global correlation value", "callable": out_callable},
    )
]
GCOR_output_spec = specs.SpecInfo(
    name="Output", fields=output_fields, bases=(specs.ShellOutSpec,)
)


class GCOR(ShellCommandTask):
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from fileformats.medimage import Nifti1
    >>> from pydra.tasks.mriqc.interfaces.transitional.gcor import GCOR

    >>> task = GCOR()
    >>> task.inputs.in_file = Nifti1.mock("func.nii")
    >>> task.inputs.mask = File.mock()
    >>> task.inputs.nfirst = 4
    >>> task.cmdline
    '@compute_gcor -nfirst 4 -input func.nii'


    """

    input_spec = GCOR_input_spec
    output_spec = GCOR_output_spec
    executable = "@compute_gcor"
