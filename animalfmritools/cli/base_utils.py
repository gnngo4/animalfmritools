from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from animalfmritools.utils.data_grabber import BidsReader

TEMPLATE_DIR = {
    "mouse": Path("/app/animalfmritools/data_template/mouse/template"),  # ABAv3
    "rat": Path("/app/animalfmritools/data_template/rat/template"),  # WHS
    "marmoset": Path("/app/animalfmritools/data_template/marmoset/template"),  # MBMv4
}
SURFACE_DIR = {
    "mouse": Path("/app/animalfmritools/data_template/mouse/surfaces/3k"),  # ABAv3
    "rat": Path("/app/animalfmritools/data_template/rat/surfaces/7k"),  # WHS
    "marmoset": Path("/app/animalfmritools/data_template/marmoset/surfaces/10k"),  # MBMv4
}


class WorkflowManager(BaseModel):
    """Model to manage workflow parameters.

    Attributes:
        sub_id (str): Subject ID.
        ses_id (str): Session ID.
        bids_dir (Path): Directory containing BIDS-formatted data.
        deriv_dir (Path): Directory to store pipeline's output.
        scratch_dir (Path): Directory for temporary files.
        anat (Dict[str, Path]): Anatomical image paths.
        bold_runs (Dict[str, List[Path]]): BOLD run paths.
        fmap_runs (Dict[str, List[Path]]): Fieldmap paths of opposite phase-encoded images.
        template (Dict[str, Path]): Template image paths.
        surfaces (Dict[str, Path]): Surface paths.
    """

    sub_id: str
    ses_id: str
    bids_dir: Path
    deriv_dir: Path
    scratch_dir: Path
    anat: Dict[str, Path]
    bold_runs: Dict[str, List[Path]]
    fmap_runs: Dict[str, List[Path]]
    template: Dict[str, Path]
    surfaces: Dict[str, Path]


def get_template_data(species_id: str) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Retrieve template and surface data for a given species.

    Args:
        species_id (str): Identifier for the species. Only "marmoset", "rat" and "mouse" are supported.

    Returns:
        Tuple[Dict[str, Path], Dict[str, Path]]: Template and surface data paths.
    """

    assert species_id in ["marmoset", "mouse", "rat"]
    template_base_dir = TEMPLATE_DIR[species_id]
    surface_base_dir = SURFACE_DIR[species_id]
    if species_id == "mouse":
        template = {
            "Base": template_base_dir / "TMBTA_space-P56_downsample2.nii.gz",
            "CSF": template_base_dir / "pipeline_vs.nii.gz",
            "Grey": template_base_dir / "pipeline_gm.nii.gz",
            "White": template_base_dir / "pipeline_wm.nii.gz",
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
    elif species_id == "rat":
        template = {
            "Base": template_base_dir / "WHS_SD_rat_T2star_v1.01_brain.nii.gz",
            "CSF": template_base_dir / "pipeline_vs.nii.gz",
            "Grey": template_base_dir / "pipeline_gm.nii.gz",
            "White": template_base_dir / "pipeline_wm.nii.gz",
        }
        surfaces = {
            "lh_midthickness": surface_base_dir / "WHS.lh.midthickness.7k.surf.gii",
            "rh_midthickness": surface_base_dir / "WHS.rh.midthickness.7k.surf.gii",
            "lh_white": surface_base_dir / "WHS.lh.white.7k.surf.gii",
            "rh_white": surface_base_dir / "WHS.rh.white.7k.surf.gii",
            "lh_pial": surface_base_dir / "WHS.lh.pial.7k.surf.gii",
            "rh_pial": surface_base_dir / "WHS.rh.pial.7k.surf.gii",
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
    use_anat_to_guide: bool = False,
    anat_contrast_type: str = "T2w",
) -> WorkflowManager:
    """Set up the workflow manager.

    Args:
        species_id (str): Identifier for the species.
        sub_id (str): Subject ID.
        ses_id (str): Session ID.
        bids_dir (Path): Directory containing BIDS-formatted data.
        deriv_dir (Path): Directory to store pipeline's output.
        scratch_dir (Path): Directory for temporary files.
        force_anat (Optional[Path]): Force the use of a specific anatomical image. (default: None)
        use_anat_to_guide (bool): Whether to use anatomical image to guide. (default: False)
        anat_contrast_type (str): Type of anatomical contrast. (default: "T2w)

    Returns:
        WorkflowManager: Instance of WorkflowManager.
    """

    bids_reader = BidsReader(bids_dir)

    template, surfaces = get_template_data(species_id)

    data = {
        "sub_id": sub_id,
        "ses_id": ses_id,
        "bids_dir": bids_dir,
        "deriv_dir": deriv_dir,
        "scratch_dir": scratch_dir,
        "anat": bids_reader.get_anat(
            sub_id,
            ses_id=ses_id,
            force_anat=force_anat,
            use_anat_to_guide=use_anat_to_guide,
            contrast_type=anat_contrast_type,
        ),
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
