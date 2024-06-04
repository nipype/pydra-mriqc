import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.diffusion.base import hmc_workflow
import pytest


logger = logging.getLogger(__name__)


def test_hmc_workflow_build():
    workflow = hmc_workflow()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_hmc_workflow_run():
    workflow = hmc_workflow()
    result = workflow(plugin="serial")
    print(result.out)
