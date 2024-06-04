import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.output import init_anat_report_wf
import pytest


logger = logging.getLogger(__name__)


def test_init_anat_report_wf_build():
    workflow = init_anat_report_wf()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_init_anat_report_wf_run():
    workflow = init_anat_report_wf()
    result = workflow(plugin="serial")
    print(result.out)
