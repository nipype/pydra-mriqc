import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import compute_iqms
import pytest


logger = logging.getLogger(__name__)


def test_compute_iqms_build():
    workflow = compute_iqms()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_compute_iqms_run():
    workflow = compute_iqms()
    result = workflow(plugin="serial")
    print(result.out)
