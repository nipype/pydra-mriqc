import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import hmc
import pytest


logger = logging.getLogger(__name__)


def test_hmc_build():
    workflow = hmc()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_hmc_run():
    workflow = hmc()
    result = workflow(plugin="serial")
    print(result.out)
