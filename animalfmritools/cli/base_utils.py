from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from animalfmritools.utils.data_grabber import BidsReader

# DATADIR = Path("/opt/animalfmritools/animalfmritools/data")
TEMPLATE_DIR = {
    "MouseABA": Path("/opt/animalfmritools/animalfmritools/data_template/MouseABA"),  # Mouse
    "MBM_v3.0.1": Path("/opt/animalfmritools/animalfmritools/data_template/MBM_v3.0.1"),  # Marmoset
}


class WorkflowManager(BaseModel):
    sub_id: str
    ses_id: str
    bids_dir: Path
    deriv_dir: Path
    scratch_dir: Path
    anat: Path
    bold_runs: Dict[str, List[Path]]
    fmap_runs: Dict[str, List[Path]]
    template: Dict[str, Path]


def get_template_data(species_id: str) -> Dict[str, Path]:
    assert species_id in ["marmoset", "mouse"]
    if species_id == "mouse":
        base_dir = TEMPLATE_DIR["MouseABA"]
        template = {
            "Base": base_dir / "TMBTA_space-P56_downsample2.nii.gz",
            "CSF": base_dir / "rois" / "pipeline_vs.nii.gz",
            "Grey": base_dir / "rois" / "pipeline_gm.nii.gz",
            "White": base_dir / "rois" / "pipeline_wm.nii.gz",
        }
    elif species_id == "marmoset":
        base_dir = TEMPLATE_DIR["MBM_v3.0.1"]
        template = {
            "Base": base_dir / "template_T2w_brain.nii.gz",
            "CSF": base_dir / "pipeline_csf.nii.gz",
            "Grey": base_dir / "pipeline_gm.nii.gz",
            "White": base_dir / "pipeline_wm.nii.gz",
        }
    else:
        raise ValueError(f"{species_id} not implemented")

    for k, v in template.items():
        assert v.exists(), f"[{k}] Path: {v} does not exist."

    return template


def setup_workflow(
    species_id: str,
    sub_id: str,
    ses_id: str,
    bids_dir: Path,
    deriv_dir: Path,
    scratch_dir: Path,
) -> WorkflowManager:
    bids_reader = BidsReader(bids_dir)
    data = {
        "sub_id": sub_id,
        "ses_id": ses_id,
        "bids_dir": bids_dir,
        "deriv_dir": deriv_dir,
        "scratch_dir": scratch_dir,
        "anat": bids_reader.get_anat(sub_id),
        "bold_runs": bids_reader.get_bold_runs(sub_id, ses_id, ignore_tasks=[]),
        "fmap_runs": bids_reader.get_fmap_runs(sub_id, ses_id),
        "template": get_template_data(species_id),
    }

    # Verbose; debug
    """
    for i, j in data.items():
        print(i,j)

    import pdb; pdb.set_trace()
    """

    return WorkflowManager(**data)
