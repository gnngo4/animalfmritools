from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.interfaces.io import ExportFile
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pydantic import BaseModel


class BaseInfo(BaseModel):
    """Base class for storing information of the BOLD run.

    Attributes:
        sub_id (str): Subject ID.
        ses_id (str): Session ID.
        task_id (str): Task ID.
        dir_id (str): Phase-encoding direction.
        run_id (str): Run ID.
    """

    sub_id: str
    ses_id: str
    task_id: str
    dir_id: str
    run_id: str


class DerivativeOutputs(BaseModel):
    """Model for storing derivative outputs of the BOLD run.

    Attributes:
        bold_preproc (Path): Minimally preprocessed BOLD data that has been normalized to template space. (.nii.gz)
        bold_preproc_dtseries (Path): Minimally preprocessed BOLD data projected on the the cortical surface of the template. (.dtseries.nii)
        bold_confounds (Path): TSV file with various regressors. (.tsv)
        bold_confounds_metadata (Path): File containing metadata of various regressors. (.json)
        bold_roi_svg (Path): Image of template normalized BOLD run with some overlays. (.svg)
        reg_from_Dbold_to_Dboldtemplate (Path): Registration from Dbold to Dboldtemplate.
    """

    bold_preproc: Path
    bold_preproc_dtseries: Path
    bold_confounds: Path
    bold_confounds_metadata: Path
    bold_roi_svg: Path
    reg_from_Dbold_to_Dboldtemplate: Path


def parse_bids_tag(stem: str, tag: str) -> str:
    """Parse a BIDS tag from the stem of a file name.

    Args:
        stem (str): The stem of the file name.
        tag (str): The BIDS tag to parse.

    Returns:
        str: The value of the specified BIDS tag in the file name stem.

    Raises:
        AssertionError: If the specified tag is not found in the stem.
    """
    assert f"{tag}-" in stem, f"{tag} not found in {stem}."
    value = stem.split(f"{tag}-")[1].split("_")[0]

    return value


def parse_bold_path(bold_path: Path) -> BaseInfo:
    """Parse BIDS information from a BOLD file path.

    Args:
        bold_path (Path): The path to the BOLD file.

    Returns:
        BaseInfo: An instance of BaseInfo containing parsed BIDS information from the file path.
    """
    stem = bold_path.stem
    info = BaseInfo(
        sub_id=parse_bids_tag(stem, "sub"),
        ses_id=parse_bids_tag(stem, "ses"),
        task_id=parse_bids_tag(stem, "task"),
        dir_id=parse_bids_tag(stem, "dir"),
        run_id=parse_bids_tag(stem, "run"),
    )
    return info


def get_source_files(base_info: BaseInfo, derivatives_dir: Path) -> DerivativeOutputs:
    """Retrieve source files and return as DerivativeOutputs object.

    Args:
        base_info (BaseInfo): Base information with subject and session IDs.
        derivatives_dir (Path): Path to the derivatives directory.

    Returns:
        DerivativeOutputs: An instance of DerivativeOutputs containing the retrieved source files.
    """

    bold_preproc = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/func/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_task-{base_info.task_id}"
        f"_dir-{base_info.dir_id}_run-{base_info.run_id}_space-template_desc-preproc_bold.nii.gz"
    )
    bold_preproc_dtseries = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/func/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_task-{base_info.task_id}"
        f"_dir-{base_info.dir_id}_run-{base_info.run_id}_space-template_desc-preproc_bold.dtseries.nii"
    )
    bold_confounds = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/func/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_task-{base_info.task_id}"
        f"_dir-{base_info.dir_id}_run-{base_info.run_id}_desc-confounds_timeseries.tsv"
    )
    bold_confounds_metadata = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/func/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_task-{base_info.task_id}"
        f"_dir-{base_info.dir_id}_run-{base_info.run_id}_desc-confounds_timeseries.json"
    )
    bold_roi_svg = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_task-{base_info.task_id}"
        f"_dir-{base_info.dir_id}_run-{base_info.run_id}_desc-confound_roi.svg"
    )

    """
    Registration is between single bold runs and PE-consistant bold referance run
    * Calculated for every bold run
    * PE-consistent bold run is a designated single bold run,
    and defined for each PE-direction
    * D = Distorted (non-SDC corrected)
    """
    reg_from_Dbold_to_Dboldtemplate = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_task-{base_info.task_id}"
        f"_dir-{base_info.dir_id}_run-{base_info.run_id}_from-Dbold_to-Dboldtemplate.svg"
    )

    outputs = DerivativeOutputs(
        bold_preproc=bold_preproc,
        bold_preproc_dtseries=bold_preproc_dtseries,
        bold_confounds=bold_confounds,
        bold_confounds_metadata=bold_confounds_metadata,
        bold_roi_svg=bold_roi_svg,
        reg_from_Dbold_to_Dboldtemplate=reg_from_Dbold_to_Dboldtemplate,
    )

    return outputs


def create_base_directories(outputs: DerivativeOutputs) -> None:
    """Create base directories for the paths in the DerivativeOutputs object.

    Args:
        outputs (DerivativeOutputs): An instance of DerivativeOutputs containing paths.

    Returns:
        None
    """
    for k, _v in outputs.__annotations__.items():
        p = getattr(outputs, k).parent
        if not p.exists():
            p.mkdir(parents=True)


def init_bold_preproc_derivatives_wf(
    outputs: DerivativeOutputs,
    name: str,
) -> Workflow:
    """Construct a workflow to manage all pipeline outputs pertaining to the BOLD run.

    Args:
        outputs (DerivativeOutputs): An instance of DerivativeOutputs containing paths.
        name (str): Name of the workflow.

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        Refer to DerivativeOutputs BaseModel for information on workflow inputs.
    """
    workflow = Workflow(name=name)

    # Create all expected parent directories found in `outputs
    create_base_directories(outputs)

    inputnode_fields = list(outputs.__annotations__.keys())
    inputnode = pe.Node(niu.IdentityInterface(fields=inputnode_fields), name="inputnode")

    for f in inputnode_fields:
        out_file = getattr(outputs, f)
        ds = pe.Node(
            ExportFile(
                out_file=out_file,
                check_extension=False,
                clobber=True,
            ),
            name=f"ds_{f}",
            run_without_submitting=True,
        )
        # fmt: off
        workflow.connect([(inputnode, ds, [(f, "in_file")])])
        # fmt: on

    return workflow
