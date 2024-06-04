import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.shared import synthstrip_wf
import pytest


logger = logging.getLogger(__name__)


def test_synthstrip_wf_build():
    workflow = synthstrip_wf(omp_nthreads=1)
    assert isinstance(workflow, Workflow)


@pytest.mark.skip(
    reason="Appropriate inputs for this workflow haven't been specified yet"
)
def test_synthstrip_wf_run():
    workflow = synthstrip_wf()
    result = workflow(plugin="serial")
    print(result.out)
