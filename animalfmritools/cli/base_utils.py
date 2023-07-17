from pydantic import BaseModel

from typing import Dict, List
from pathlib import Path

from animalfmritools.utils.data_grabber import BidsReader

DATADIR = Path("/opt/animalfmritools/animalfmritools/data")


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


def get_template_data(base_dir: Path = DATADIR) -> Dict[str, Path]:
    return {
        "Base": base_dir / "TMBTA_Brain_Template.nii.gz",
        "CSF": base_dir / "TMBTA_Ventricles.nii.gz",
        "Grey": base_dir / "TMBTA_Grey.nii.gz",
        "White": base_dir / "TMBTA_White_eroF.nii.gz",
    }


def setup_workflow(
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
        "template": get_template_data(),
    }

    # Verbose; debug
    """
    for i, j in data.items():
        print(i,j)

    import pdb; pdb.set_trace()
    """

    return WorkflowManager(**data)
