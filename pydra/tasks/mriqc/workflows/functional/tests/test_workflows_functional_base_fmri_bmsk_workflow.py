import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import fmri_bmsk_workflow
import pytest


logger = logging.getLogger(__name__)


def test_fmri_bmsk_workflow_build():
    workflow = fmri_bmsk_workflow()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_fmri_bmsk_workflow_run():
    workflow = fmri_bmsk_workflow()
    result = workflow(plugin="serial")
    print(result.out)
