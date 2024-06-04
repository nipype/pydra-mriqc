import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import airmsk_wf
import pytest


logger = logging.getLogger(__name__)


def test_airmsk_wf_build():
    workflow = airmsk_wf()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_airmsk_wf_run():
    workflow = airmsk_wf()
    result = workflow(plugin="serial")
    print(result.out)
