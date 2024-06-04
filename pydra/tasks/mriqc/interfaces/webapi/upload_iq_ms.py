import attrs
import logging
from pathlib import Path
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"api_id": ty.Any}})
def UploadIQMs(
    in_iqms: dict = attrs.NOTHING,
    endpoint: str = attrs.NOTHING,
    auth_token: str = attrs.NOTHING,
    email: str = attrs.NOTHING,
    strict: bool = False,
) -> ty.Any:
    """
    Examples
    -------

    >>> from pydra.tasks.mriqc.interfaces.webapi.upload_iq_ms import UploadIQMs

    """
    api_id = attrs.NOTHING
    email = None
    if email is not attrs.NOTHING:
        email = email

    api_id = None

    response = upload_qc_metrics(
        in_iqms,
        endpoint=endpoint,
        auth_token=auth_token,
        email=email,
    )

    try:
        api_id = response.json()["_id"]
    except (AttributeError, KeyError, ValueError):

        errmsg = (
            "QC metrics upload failed to create an ID for the record "
            f"uplOADED. rEsponse from server follows: {response.text}"
        )
        logger.warning(errmsg)

    if response.status_code == 201:
        logger.info('QC metrics successfully uploaded.')

    errmsg = "QC metrics failed to upload. Status %d: %s" % (
        response.status_code,
        response.text,
    )
    logger.warning(errmsg)
    if strict:
        raise RuntimeError(response.text)

    return api_id


# Nipype methods converted into functions


def _hashfields(data):

    from hashlib import sha256

    for name in HASH_BIDS:
        if name in data:
            data[name] = sha256(data[name].encode()).hexdigest()
    return data


def upload_qc_metrics(in_iqms, endpoint=None, email=None, auth_token=None):
    """
    Upload qc metrics to remote repository.

    :param str in_iqms: Path to the qc metric json file as a string
    :param str webapi_url: the protocol (either http or https)
    :param str email: email address to be included with the metric submission
    :param str auth_token: authentication token

    :return: either the response object if a response was successfully sent
             or it returns the string "No Response"
    :rtype: object


    """
    from copy import deepcopy
    from json import dumps, loads
    from pathlib import Path
    import requests

    if not endpoint or not auth_token:
        # If endpoint unknown, do not even report what happens to the token.
        errmsg = "Unknown API endpoint" if not endpoint else "Authentication failed."
        return Bunch(status_code=1, text=errmsg)
    in_data = loads(Path(in_iqms).read_text())
    # Extract metadata and provenance
    meta = in_data.pop("bids_meta")
    # For compatibility with WebAPI. Should be rolled back to int
    if meta.get("run_id", None) is not None:
        meta["run_id"] = "%d" % meta.get("run_id")
    prov = in_data.pop("provenance")
    # At this point, data should contain only IQMs
    data = deepcopy(in_data)
    # Check modality
    modality = meta.get("modality", "None")
    if modality not in ("T1w", "bold", "T2w"):
        errmsg = (
            'Submitting to MRIQCWebAPI: image modality should be "bold", "T1w", or "T2w", '
            '(found "%s")' % modality
        )
        return Bunch(status_code=1, text=errmsg)
    # Filter metadata values that aren't in whitelist
    data["bids_meta"] = {k: meta[k] for k in META_WHITELIST if k in meta}
    # Filter provenance values that aren't in whitelist
    data["provenance"] = {k: prov[k] for k in PROV_WHITELIST if k in prov}
    # Hash fields that may contain personal information
    data["bids_meta"] = _hashfields(data["bids_meta"])
    if email:
        data["provenance"]["email"] = email
    headers = {"Authorization": auth_token, "Content-Type": "application/json"}
    start_message = 'MRIQC Web API: submitting to <{url}>'.format(url=endpoint)
    logger.info(start_message)
    try:
        # if the modality is bold, call "bold" endpoint
        response = requests.post(
            f"{endpoint}/{modality}",
            headers=headers,
            data=dumps(data),
            timeout=15,
        )
    except requests.ConnectionError as err:
        errmsg = (
            "QC metrics failed to upload due to connection error shown below:\n%s" % err
        )
        return Bunch(status_code=1, text=errmsg)
    return response


HASH_BIDS = ["subject_id", "session_id"]

META_WHITELIST = [
    "AccelNumReferenceLines",
    "AccelerationFactorPE",
    "AcquisitionMatrix",
    "CogAtlasID",
    "CogPOID",
    "CoilCombinationMethod",
    "ContrastBolusIngredient",
    "ConversionSoftware",
    "ConversionSoftwareVersion",
    "DelayTime",
    "DeviceSerialNumber",
    "EchoTime",
    "EchoTrainLength",
    "EffectiveEchoSpacing",
    "FlipAngle",
    "GradientSetType",
    "HardcopyDeviceSoftwareVersion",
    "ImageType",
    "ImagingFrequency",
    "InPlanePhaseEncodingDirection",
    "InstitutionAddress",
    "InstitutionName",
    "Instructions",
    "InversionTime",
    "MRAcquisitionType",
    "MRTransmitCoilSequence",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturersModelName",
    "MatrixCoilMode",
    "MultibandAccelerationFactor",
    "NumberOfAverages",
    "NumberOfPhaseEncodingSteps",
    "NumberOfVolumesDiscardedByScanner",
    "NumberOfVolumesDiscardedByUser",
    "NumberShots",
    "ParallelAcquisitionTechnique",
    "ParallelReductionFactorInPlane",
    "PartialFourier",
    "PartialFourierDirection",
    "PatientPosition",
    "PercentPhaseFieldOfView",
    "PercentSampling",
    "PhaseEncodingDirection",
    "PixelBandwidth",
    "ProtocolName",
    "PulseSequenceDetails",
    "PulseSequenceType",
    "ReceiveCoilName",
    "RepetitionTime",
    "ScanOptions",
    "ScanningSequence",
    "SequenceName",
    "SequenceVariant",
    "SliceEncodingDirection",
    "SoftwareVersions",
    "TaskDescription",
    "TaskName",
    "TotalReadoutTime",
    "TotalScanTimeSec",
    "TransmitCoilName",
    "VariableFlipAngleFlag",
    "acq_id",
    "modality",
    "run_id",
    "subject_id",
    "task_id",
    "session_id",
]

PROV_WHITELIST = ["version", "md5sum", "software", "settings"]
