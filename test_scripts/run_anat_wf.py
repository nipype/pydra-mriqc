from fileformats.medimage import NiftiGzX, T1Weighted
import logging
from pathlib import Path
from pydra.tasks.mriqc.workflows.anatomical.base import anat_qc_workflow

log_file = Path("/Users/tclose/Data/pydra-mriqc-test.log")
log_file.unlink(missing_ok=True)

pydra_logger = logging.getLogger("pydra")
pydra_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_file))
pydra_logger.addHandler(file_handler)
pydra_logger.addHandler(logging.StreamHandler())

workflow = anat_qc_workflow(in_file=NiftiGzX[T1Weighted].sample(), modality="T1w")
workflow.cache_dir = "/Users/tclose/Data/pydra-mriqc-test-cache"
result = workflow(plugin="serial")
print(result.out)
