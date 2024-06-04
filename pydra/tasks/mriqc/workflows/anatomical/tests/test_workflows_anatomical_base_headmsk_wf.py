import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import headmsk_wf
import pytest


logger = logging.getLogger(__name__)


def test_headmsk_wf_build():
    workflow = headmsk_wf()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_headmsk_wf_run():
    workflow = headmsk_wf()
    result = workflow(plugin="serial")
    print(result.out)
