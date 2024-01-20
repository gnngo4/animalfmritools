from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.interfaces.io import ExportFile
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pydantic import BaseModel


class BaseInfo(BaseModel):
    sub_id: str
    ses_id: str
    task_id: str
    dir_id: str
    run_id: str


class DerivativeOutputs(BaseModel):
    bold_preproc: Path
    bold_preproc_dtseries: Path
    bold_confounds: Path
    bold_confounds_metadata: Path
    bold_roi_svg: Path
    reg_from_Dbold_to_Dboldtemplate: Path


def parse_bids_tag(stem: str, tag: str) -> str:
    assert f"{tag}-" in stem, f"{tag} not found in {stem}."
    value = stem.split(f"{tag}-")[1].split("_")[0]

    return value


def parse_bold_path(bold_path: Path) -> BaseInfo:
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
    for k, _v in outputs.__annotations__.items():
        p = getattr(outputs, k).parent
        if not p.exists():
            p.mkdir(parents=True)


def init_bold_preproc_derivatives_wf(
    outputs: DerivativeOutputs,
    name: str,
) -> Workflow:
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
