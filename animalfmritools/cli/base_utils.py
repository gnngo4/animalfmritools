from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from animalfmritools.utils.data_grabber import BidsReader

TEMPLATE_DIR = {
    "mouse": Path("/opt/animalfmritools/animalfmritools/data_template/mouse/template"),  # ABAv3
    "marmoset": Path("/opt/animalfmritools/animalfmritools/data_template/marmoset/template"),  # MBMv4
}
SURFACE_DIR = {
    "mouse": Path("/opt/animalfmritools/animalfmritools/data_template/mouse/surfaces/3k"),  # ABAv3
    "marmoset": Path("/opt/animalfmritools/animalfmritools/data_template/marmoset/surfaces/10k"),  # MBMv4
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
    surfaces: Dict[str, Path]


def get_template_data(species_id: str) -> Dict[str, Path]:
    assert species_id in ["marmoset", "mouse"]
    template_base_dir = TEMPLATE_DIR[species_id]
    surface_base_dir = SURFACE_DIR[species_id]
    if species_id == "mouse":
        template = {
            "Base": template_base_dir / "TMBTA_space-P56_downsample2.nii.gz",
            "CSF": template_base_dir / "rois" / "pipeline_vs.nii.gz",
            "Grey": template_base_dir / "rois" / "pipeline_gm.nii.gz",
            "White": template_base_dir / "rois" / "pipeline_wm.nii.gz",
        }
        surfaces = {
            "lh_midthickness": surface_base_dir / "ABAv3.lh.midthickness.3k.surf.gii",
            "rh_midthickness": surface_base_dir / "ABAv3.rh.midthickness.3k.surf.gii",
            "lh_white": surface_base_dir / "ABAv3.lh.white.3k.surf.gii",
            "rh_white": surface_base_dir / "ABAv3.rh.white.3k.surf.gii",
            "lh_pial": surface_base_dir / "ABAv3.lh.pial.3k.surf.gii",
            "rh_pial": surface_base_dir / "ABAv3.rh.pial.3k.surf.gii",
            "lh_cortex": surface_base_dir / "cortex.lh.func.gii",
            "rh_cortex": surface_base_dir / "cortex.rh.func.gii",
        }
    elif species_id == "marmoset":
        template = {
            "Base": template_base_dir / "template_T2w_brain.nii.gz",
            "CSF": template_base_dir / "pipeline_csf.nii.gz",
            "Grey": template_base_dir / "pipeline_gm.nii.gz",
            "White": template_base_dir / "pipeline_wm.nii.gz",
        }
        surfaces = {
            "lh_midthickness": surface_base_dir / "surfFS.lh.graymid.10k.surf.gii",
            "rh_midthickness": surface_base_dir / "surfFS.rh.graymid.10k.surf.gii",
            "lh_white": surface_base_dir / "surfFS.lh.white.10k.surf.gii",
            "rh_white": surface_base_dir / "surfFS.rh.white.10k.surf.gii",
            "lh_pial": surface_base_dir / "surfFS.lh.pial.10k.surf.gii",
            "rh_pial": surface_base_dir / "surfFS.rh.pial.10k.surf.gii",
            "lh_cortex": surface_base_dir / "cortex.lh.func.gii",
            "rh_cortex": surface_base_dir / "cortex.rh.func.gii",
        }
    else:
        raise ValueError(f"{species_id} not implemented")

    for k, v in template.items():
        assert v.exists(), f"[{k}] Path: {v} does not exist."

    for k, v in surfaces.items():
        assert v.exists(), f"[{k}] Path: {v} does not exist."

    return template, surfaces


def setup_workflow(
    species_id: str,
    sub_id: str,
    ses_id: str,
    bids_dir: Path,
    deriv_dir: Path,
    scratch_dir: Path,
    force_anat: Optional[Path] = None,
    anat_contrast_type: str = "T2w",
) -> WorkflowManager:
    bids_reader = BidsReader(bids_dir)

    template, surfaces = get_template_data(species_id)

    data = {
        "sub_id": sub_id,
        "ses_id": ses_id,
        "bids_dir": bids_dir,
        "deriv_dir": deriv_dir,
        "scratch_dir": scratch_dir,
        "anat": bids_reader.get_anat(sub_id, ses_id=ses_id, force_anat=force_anat, contrast_type=anat_contrast_type),
        "bold_runs": bids_reader.get_bold_runs(sub_id, ses_id, ignore_tasks=[]),
        "fmap_runs": bids_reader.get_fmap_runs(sub_id, ses_id),
        "template": template,
        "surfaces": surfaces,
    }

    # Verbose; debug
    """
    for i, j in data.items():
        print(i,j)

    import pdb; pdb.set_trace()
    """

    return WorkflowManager(**data)
