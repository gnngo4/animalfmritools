from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

REVERSE_PE_MAPPING = {
    "dir-AP": "dir-PA",
    "dir-PA": "dir-AP",
    "dir-LR": "dir-RL",
    "dir-RL": "dir-LR",
    "dir-IS": "dir-SI",
    "dir-SI": "dir-IS",
}

PE_DIR_FLIP: Dict[str, bool] = {
    "dir-AP": False,
    "dir-PA": True,
    "dir-LR": False,
    "dir-RL": True,
    "dir-IS": False,
    "dir-SI": True,
}

PE_DIR_SCHEMA: Dict[str, List[str]] = {
    "dir-AP": [],
    "dir-PA": [],
    "dir-LR": [],
    "dir-RL": [],
    "dir-IS": [],
    "dir-SI": [],
}


class BidsReaderInput(BaseModel):
    """Input model for BIDS reader."""

    bids_dir: Path


class BidsReader(BidsReaderInput):
    """BIDS reader class.

    Args:
        bids_dir (Path): Path to the BIDS directory.

    Attributes:
        bids_dir (Path): Path to the BIDS directory.

    Methods:
        get_subjects: Retrieve a list of subjects.
        get_sessions: Retrieve a list of sessions for a given subject.
        get_anat: Retrieve anatomical images.
        get_bold_runs: Retrieve BOLD runs.
        get_fmap_runs: Retrieve field map runs.
    """

    def __init__(self, bids_dir: Path):
        """Initialize BIDS reader.

        Args:
            bids_dir (Path): Path to the BIDS directory.
        """
        input_data = BidsReaderInput(bids_dir=bids_dir)
        super().__init__(**input_data.dict())

    def get_subjects(self) -> List[str]:
        """Retrieve a list of subjects.

        Returns:
            List[str]: List of subject IDs.
        """
        subdirs = []
        for i in self.bids_dir.iterdir():
            if i.is_dir() and i.name.startswith("sub-"):
                subdirs.append(i.stem)
        subdirs.sort()

        return subdirs

    def get_sessions(self, sub_id: str) -> List[str]:
        """Retrieve a list of sessions for a given subject.

        Returns:
            List[str]: List of sessions IDs.
        """
        subdirs = []
        sub_dir = Path(f"{self.bids_dir}/{self._process_dir(sub_id,'sub')}")
        assert sub_dir.exists(), f"Directory [{sub_dir}] does not exist."

        for i in sub_dir.iterdir():
            if i.is_dir() and i.name.startswith("ses-"):
                subdirs.append(i.stem)
        subdirs.sort()

        return subdirs

    def get_anat(
        self,
        sub_id: str,
        ses_id: Optional[str] = None,
        force_anat: Optional[Path] = None,
        use_anat_to_guide: bool = False,
        contrast_type: str = "T2w",
    ) -> Dict[str, Path]:
        """Retrieve anatomical images.

        Args:
            sub_id (str): Subject ID.
            ses_id (Optional[str]): Session ID. (default: None)
            force_anat (Optional[Path]): Force the use of a specific anatomical image. (default: None)
            use_anat_to_guide (bool): Setting to True will use the anatomical image to guide the alignment. (default: False)
            contrast_type (str): Type of contrast (i.e., this is denoted by the suffix of a BIDS-formatted NIFTI). (default: "T2w")

        Returns:
            Dict[str, Path]: Dictionary containing the path to the anatomical images.
        """
        anat_native = self._find_last_t2w_run(sub_id, ses_id, contrast_type)

        if force_anat:
            assert force_anat.exists(), f"--force_anat was set, but {force_anat} does not exist."
            if use_anat_to_guide:
                print(f"Warning: Alignment order: {anat_native} > {force_anat} > TEMPLATE.")
                return {"anat_native": anat_native, "anat_template": force_anat}
            else:
                print(f"Warning: Using {force_anat} as the anatomical image.")
                return {"anat_template": force_anat}

        return {"anat_template": anat_native}

    def get_bold_runs(
        self, sub_id: str, ses_id: str, ignore_tasks: Optional[List[str]] = None
    ) -> Dict[str, List[Path]]:
        """Retrieve BOLD runs.

        Args:
            sub_id (str): Subject ID.
            ses_id (str): Session ID.
            ignore_tasks (Optional[List[str]]): List of task IDs to ignore. (default: None)

        Returns:
            Dict[str, List[Path]]: Dictionary containing BOLD runs.
        """

        if ignore_tasks is None:
            ignore_tasks = []
        ignore_tasks = [f"task-{task_id}" if not task_id.startswith("task-") else task_id for task_id in ignore_tasks]

        runs_dict = deepcopy(PE_DIR_SCHEMA)

        sub_ses_func_dir = (
            Path(self.bids_dir) / self._process_dir(sub_id, "sub") / self._process_dir(ses_id, "ses") / "func"
        )
        assert sub_ses_func_dir.exists(), f"Directory [{sub_ses_func_dir}] does not exist."

        for i in sub_ses_func_dir.iterdir():
            if any(ignore_task in i.stem for ignore_task in ignore_tasks):
                continue

            if not i.is_dir() and i.name.endswith("_bold.nii.gz"):
                _dir = i.stem.split("_dir-")[-1].split("_")[0]
                _dir = f"dir-{_dir}"
                assert _dir in runs_dict.keys(), f"{_dir} is not supported."
                runs_dict[_dir].append(i)

        # sort
        for _, runs in runs_dict.items():
            runs.sort()

        # Remove part-phase of a run
        filtered_dict = {}
        for k, runs in runs_dict.items():
            filtered_list = self._remove_phase_parts(runs)
            filtered_dict[k] = filtered_list

        return filtered_dict

    def get_fmap_runs(
        self,
        sub_id: str,
        ses_id: str,
    ) -> Dict[str, List[Path]]:
        """Retrieve field map runs.

        Args:
            sub_id (str): Subject ID.
            ses_id (str): Session ID.

        Returns:
            Dict[str, List[Path]]: Dictionary containing field map runs.
        """

        runs_dict = deepcopy(PE_DIR_SCHEMA)
        sub_ses_fmap_dir = (
            Path(self.bids_dir) / self._process_dir(sub_id, "sub") / self._process_dir(ses_id, "ses") / "fmap"
        )
        # assert sub_ses_fmap_dir.exists(), f"Directory [{sub_ses_fmap_dir}] does not exist."

        if sub_ses_fmap_dir.exists():
            for i in sub_ses_fmap_dir.iterdir():
                if not i.is_dir() and i.name.endswith("_epi.nii.gz"):
                    _dir = i.stem.split("_dir-")[-1].split("_")[0]
                    _dir = f"dir-{_dir}"
                    assert _dir in runs_dict.keys(), f"{_dir} is not supported."
                    runs_dict[_dir].append(i)
            # sort
            for _k, runs in runs_dict.items():
                runs.sort()

        return runs_dict

    def _find_last_t2w_run(
        self,
        sub_id: str,
        ses_id: Optional[str] = None,
        contrast_type: str = "T2w",
    ) -> Path:
        """Find the last anatomical run.

        If more anatomicals are acquired, it is assumed that these are of higher quality.

        Args:
            sub_id (str): Subject ID.
            ses_id (Optional[str]): Session ID. (default: None)
            contrast_type (str): Type of contrast (i.e., this is denoted by the suffix of a BIDS-formatted NIFTI). (default: "T2w")

        Returns:
            Path: Path to the last T2w run.
        """
        # If multiple T2w runs detected, grab the last one
        runs: List[Path] = []
        if ses_id is not None:
            # Find a T2w given a subject and session ID
            sub_ses_anat_dir = (
                Path(self.bids_dir) / self._process_dir(sub_id, "sub") / self._process_dir(ses_id, "ses") / "anat"
            )
            for i in sub_ses_anat_dir.iterdir():
                if not i.is_dir() and i.name.endswith(f"_{contrast_type}.nii.gz"):
                    runs.append(i)
            if runs == 0:
                print(
                    f"Warning: No runs detected in {sub_ses_anat_dir}.\nSearching for an anatomical across all sessions."
                )
                # Find a T2w given a subject ID
                runs = self._find_t2w_across_sessions(sub_id, contrast_type=contrast_type)
        else:
            # Find a T2w given a subject ID
            runs = self._find_t2w_across_sessions(sub_id, contrast_type=contrast_type)

        n_runs = len(runs)
        assert n_runs > 0, "Warning: No runs were detected.\nExiting."

        if n_runs > 1:
            brainmask_runs = [_run for _run in runs if "desc-brainmask" in str(_run)]
            if len(brainmask_runs) > 1:
                print(f"Warning: Multiple brainmasks were detected, using {brainmask_runs[-1].stem}")
                return brainmask_runs[-1]
            elif len(brainmask_runs) == 1:
                print(f"Warning: Using a brainmask : {brainmask_runs[0].stem}")
                return brainmask_runs[-1]
            else:
                pass

            print(f"Warning: Multiple runs were detected, using {runs[-1].stem}")

        return runs[-1]

    def _find_t2w_across_sessions(self, sub_id: str, contrast_type: str = "T2w") -> List[Path]:
        """Find all anatomicals across sessions.

        Args:
            sub_id (str): Subject ID.
            contrast_type (str): Type of contrast (i.e., this is denoted by the suffix of a BIDS-formatted NIFTI). (default: "T2w")

        Returns:
            List[Path]: List of paths to T2w runs.
        """

        runs = []
        ses_ids = self.get_sessions(sub_id)
        for _ses_id in ses_ids:
            sub_ses_anat_dir = (
                Path(self.bids_dir) / self._process_dir(sub_id, "sub") / self._process_dir(_ses_id, "ses") / "anat"
            )
            for i in sub_ses_anat_dir.iterdir():
                if not i.is_dir() and i.name.endswith(f"_{contrast_type}.nii.gz"):
                    runs.append(i)

        return runs

    def _remove_phase_parts(self, bold_list: List[Path]) -> List[Path]:
        """Remove phase parts from BOLD runs.

        Args:
            bold_list (List[Path]): List of BOLD runs.

        Returns:
            List[Path]: List of BOLD runs with phase parts removed.
        """
        return [bold_path for bold_path in bold_list if "_part-phase_" not in str(bold_path)]

    def _process_dir(self, entry, prefix: str) -> str:
        if not entry.startswith(f"{prefix}-"):
            entry = f"{prefix}-{entry}"

        return entry
