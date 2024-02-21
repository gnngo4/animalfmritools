from pathlib import Path
from typing import Dict, List

from nipype.interfaces import utility as niu
from nipype.interfaces.io import ExportFile
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pydantic import BaseModel


class BaseInfo(BaseModel):
    """Base class for storing of the processed dataset.

    Attributes:
        sub_id (str): Subject ID.
        ses_id (str): Session ID.
    """

    sub_id: str
    ses_id: str


class DerivativeOutputs(BaseModel, extra="allow"):
    """Model for storing derivative outputs

    Attributes:
        reg_from_UDbold_to_UDboldtemplate (Dict[str, Path]): Mapping of paths representing the registration from UDbold to UDboldtemplate.
        reg_from_Dboldtemplate_to_anat (Path): Registration from Dboldtemplate to anatomical.
        boldref_from_Dboldtemplate_to_anat (Path): BOLD reference of Dboldtemplate in anatomical space.
        boldref_brainmask_from_Dboldtemplate_to_anat (Path): Brainmask of the BOLD reference in anatomical space.
        anat_brainmask_from_Dboldtemplate_to_anat (Path): Brainmask of anatomical in anatomical space.
        reg_from_UDboldtemplate_to_anat (Path): Registration from UDboldtemplate to anatomical.
        boldref_from_UDboldtemplate_to_anat (Path): BOLD reference of UDboldtemplate in anatomical space.
        boldref_brainmask_from_UDboldtemplate_to_anat (Path): Brainmask of the BOLD reference in anatomical space.
        anat_brainmask_from_UDboldtemplate_to_anat (Path): Brainmask of anatomical in anatomical space.
        reg_from_anatnative_to_anat_init (Path): Initial registration from native anatomical space to the anatomical.
        reg_from_anatnative_to_anat (Path): Registration from native anatomical space to the anatomical.
        reg_from_anat_to_template_init (Path): Initial registration from anatomical to the template.
        reg_from_anat_to_template (Path): Initial registration from anatomical to the template.
        anat_brainmask (Path): Anatomical brainmask.
    """

    reg_from_UDbold_to_UDboldtemplate: Dict[str, Path]
    reg_from_Dboldtemplate_to_anat: Path
    boldref_from_Dboldtemplate_to_anat: Path
    boldref_brainmask_from_Dboldtemplate_to_anat: Path
    anat_brainmask_from_Dboldtemplate_to_anat: Path
    reg_from_UDboldtemplate_to_anat: Path
    boldref_from_UDboldtemplate_to_anat: Path
    boldref_brainmask_from_UDboldtemplate_to_anat: Path
    anat_brainmask_from_UDboldtemplate_to_anat: Path
    reg_from_anatnative_to_anat_init: Path
    reg_from_anatnative_to_anat: Path
    reg_from_anat_to_template_init: Path
    reg_from_anat_to_template: Path
    anat_brainmask: Path

    def expand_reg_from_UDbold_to_UDboldtemplate(self):
        """Expand 'reg_from_UDbold_to_UDboldtemplate' attribute.

        This method iterates over the items in 'reg_from_UDbold_to_UDboldtemplate' dictionary,
        sets new attributes with expanded names, and then deletes the original attribute.
        """
        for from_dir, svg_out in self.reg_from_UDbold_to_UDboldtemplate.items():
            setattr(self, f"reg_from_UDbold{from_dir}_to_UDboldtemplate", svg_out)

        del self.reg_from_UDbold_to_UDboldtemplate


def load_base(sub_id: str, ses_id: str) -> BaseInfo:
    """Load base information and return as a BaseInfo object

    Args:
        sub_id (str): Subject ID.
        ses_id (str): Session ID.

    Returns:
        BaseInfo: An instance of BaseInfo containing the provided subject and session IDS.
    """
    info = BaseInfo(sub_id=sub_id, ses_id=ses_id)
    return info


def get_source_files(
    base_info: BaseInfo,
    derivatives_dir: Path,
    to_dir: str,
    from_dirs: List[str],
    anat_contrast: str,
) -> DerivativeOutputs:
    """Retrieve source files and return as DerivativeOutputs object.

    Args:
        base_info (BaseInfo): Base information with subject and session IDs.
        derivatives_dir (Path): Path to the derivatives directory.
        to_dir (str): The primary phase-encoding direction to be registered to the anatomical image.
        from_dirs (List[str]): List of all detected phase-encoding directions.
        anat_contrast (str): Contrast of the anatomical image.

    Returns:
        DerivativeOutputs: An instance of DerivativeOutputs containing the retrieved source files.
    """

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
    boldref_from_Dboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/reg_debug/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_Dboldtemplate.nii.gz"
    )
    boldref_brainmask_from_Dboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/reg_debug/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_desc-brainmask_Dboldtemplate.nii.gz"
    )
    anat_brainmask_from_Dboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/reg_debug/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_desc-brainmask_{anat_contrast}.nii.gz"
    )
    reg_from_UDboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-UDboldtemplate_to-anat.svg"
    )
    boldref_from_UDboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/reg_debug/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_UDboldtemplate.nii.gz"
    )
    boldref_brainmask_from_UDboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/reg_debug/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_desc-brainmask_UDboldtemplate.nii.gz"
    )
    anat_brainmask_from_UDboldtemplate_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/reg_debug/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_desc-brainmask_{anat_contrast}.nii.gz"
    )
    reg_from_anatnative_to_anat_init = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-anatnative_to-anat_init.svg"
    )
    reg_from_anatnative_to_anat = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-anatnative_to-anat.svg"
    )
    reg_from_anat_to_template_init = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-anat_to-template_init.svg"
    )
    reg_from_anat_to_template = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/figures/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_from-anat_to-template.svg"
    )
    anat_brainmask = Path(
        f"{derivatives_dir}/sub-{base_info.sub_id}/ses-{base_info.ses_id}/anat/"
        f"sub-{base_info.sub_id}_ses-{base_info.ses_id}_desc-brainmask_{anat_contrast}.nii.gz"
    )

    outputs = DerivativeOutputs(
        reg_from_UDbold_to_UDboldtemplate=reg_from_UDbold_to_UDboldtemplate,
        reg_from_Dboldtemplate_to_anat=reg_from_Dboldtemplate_to_anat,
        boldref_from_Dboldtemplate_to_anat=boldref_from_Dboldtemplate_to_anat,
        boldref_brainmask_from_Dboldtemplate_to_anat=boldref_brainmask_from_Dboldtemplate_to_anat,
        anat_brainmask_from_Dboldtemplate_to_anat=anat_brainmask_from_Dboldtemplate_to_anat,
        reg_from_UDboldtemplate_to_anat=reg_from_UDboldtemplate_to_anat,
        boldref_from_UDboldtemplate_to_anat=boldref_from_UDboldtemplate_to_anat,
        boldref_brainmask_from_UDboldtemplate_to_anat=boldref_brainmask_from_UDboldtemplate_to_anat,
        anat_brainmask_from_UDboldtemplate_to_anat=anat_brainmask_from_UDboldtemplate_to_anat,
        reg_from_anatnative_to_anat_init=reg_from_anatnative_to_anat_init,
        reg_from_anatnative_to_anat=reg_from_anatnative_to_anat,
        reg_from_anat_to_template_init=reg_from_anat_to_template_init,
        reg_from_anat_to_template=reg_from_anat_to_template,
        anat_brainmask=anat_brainmask,
    )
    outputs.expand_reg_from_UDbold_to_UDboldtemplate()

    return outputs


def create_base_directories(outputs: DerivativeOutputs) -> None:
    """Create base directories for the paths in the DerivativeOutputs object.

    Args:
        outputs (DerivativeOutputs): An instance of DerivativeOutputs containing paths.

    Returns:
        None
    """
    for _k, v in outputs.dict().items():
        p = v.parent
        if not p.exists():
            p.mkdir(parents=True)


def init_base_preproc_derivatives_wf(
    outputs: DerivativeOutputs,
    name: str,
    no_sdc: bool = False,
    use_anat_to_guide: bool = False,
) -> Workflow:
    """Construct a workflow to manage all pipeline outputs.

    Args:
        outputs (DerivativeOutputs): An instance of DerivativeOutputs containing paths.
        name (str): Name of the workflow.
        no_sdc (bool): If True, indicates no SDC is performed, and certain outputs are removed accordingly. (default: False)
        use_anat_to_guide (bool): If True, indicates an additional native anatomical is added to the workflow, and certain outputs are removed accordingly. (default: False)

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        Refer to DerivativeOutputs BaseModel for information on workflow inputs.
    """

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
            if f in [
                "reg_from_Dboldtemplate_to_anat",
                "boldref_from_Dboldtemplate_to_anat",
                "boldref_brainmask_from_Dboldtemplate_to_anat",
                "anat_brainmask_from_Dboldtemplate_to_anat",
            ]:
                continue

        # Add `use_anat_to_guide` heuristic
        if not use_anat_to_guide:
            if "anatnative" in f:
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
