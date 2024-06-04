import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import spatial_normalization
import pytest


logger = logging.getLogger(__name__)


def test_spatial_normalization_build():
    workflow = spatial_normalization()
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_spatial_normalization_run():
    workflow = spatial_normalization()
    result = workflow(plugin="serial")
    print(result.out)
