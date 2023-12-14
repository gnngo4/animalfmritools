from pathlib import Path
from typing import Dict, List

from nipype.interfaces import utility as niu
from nipype.interfaces.io import ExportFile
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pydantic import BaseModel


class BaseInfo(BaseModel):
    sub_id: str
    ses_id: str


class DerivativeOutputs(BaseModel, extra="allow"):
    # reg_Dboldtemplate_sdc_warp: Path
    reg_from_UDbold_to_UDboldtemplate: Dict[str, Path]
    reg_from_Dboldtemplate_to_anat: Path
    reg_from_UDboldtemplate_to_anat: Path
    reg_from_anat_to_template_init: Path
    reg_from_anat_to_template: Path

    def expand_reg_from_UDbold_to_UDboldtemplate(self):
        for from_dir, svg_out in self.reg_from_UDbold_to_UDboldtemplate.items():
            setattr(self, f"reg_from_UDbold{from_dir}_to_UDboldtemplate", svg_out)

        del self.reg_from_UDbold_to_UDboldtemplate


def load_base(sub_id: str, ses_id: str) -> BaseInfo:
    info = BaseInfo(sub_id=sub_id, ses_id=ses_id)
    return info


def get_source_files(
    base_info: BaseInfo,
    derivatives_dir: Path,
    to_dir: str,
    from_dirs: List[str],
) -> DerivativeOutputs:
    reg_from_UDbold_to_UDboldtemplate = {}
    for from_dir in from_dirs:
        if from_dir == to_dir:
            continue
        _reg = Path(
            f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
            f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-UDbold{from_dir}_to-UDboldtemplate.svg"
        )
        reg_from_UDbold_to_UDboldtemplate[from_dir] = _reg
    reg_from_Dboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-Dboldtemplate_to-anat.svg"
    )
    reg_from_UDboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-UDboldtemplate_to-anat.svg"
    )
    reg_from_anat_to_template_init = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-anat_to-template_init.svg"
    )
    reg_from_anat_to_template = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-anat_to-template.svg"
    )

    outputs = DerivativeOutputs(
        reg_from_UDbold_to_UDboldtemplate=reg_from_UDbold_to_UDboldtemplate,
        reg_from_Dboldtemplate_to_anat=reg_from_Dboldtemplate_to_anat,
        reg_from_UDboldtemplate_to_anat=reg_from_UDboldtemplate_to_anat,
        reg_from_anat_to_template_init=reg_from_anat_to_template_init,
        reg_from_anat_to_template=reg_from_anat_to_template,
    )
    outputs.expand_reg_from_UDbold_to_UDboldtemplate()

    return outputs


def create_base_directories(outputs: DerivativeOutputs) -> None:
    for _k, v in outputs.dict().items():
        p = v.parent
        if not p.exists():
            p.mkdir(parents=True)


def init_base_preproc_derivatives_wf(
    outputs: DerivativeOutputs,
    name: str,
    no_sdc: bool = False,
) -> Workflow:
    workflow = Workflow(name=name)

    # Create all expected parent directories found in `outputs
    create_base_directories(outputs)

    inputnode_fields = list(outputs.dict().keys())
    inputnode = pe.Node(niu.IdentityInterface(fields=inputnode_fields), name="inputnode")

    for f in inputnode_fields:
        # Add `no_sdc` heuristic
        if no_sdc:
            if "UDbold" in f:
                continue
        else:
            if f == "reg_from_Dboldtemplate_to_anat":
                continue

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
