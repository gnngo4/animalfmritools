from pathlib import Path

from typing import List, Dict, Optional
from pydantic import BaseModel

from copy import deepcopy

REVERSE_PE_MAPPING = {
    'dir-fwd': 'dir-rev', 
    'dir-rev': 'dir-fwd',
    'dir-AP': 'dir-PA',
    'dir-PA': 'dir-AP',
}

PE_DIR_FLIP: Dict[str, bool] = {
    'dir-AP': False,
    'dir-PA': True,
    'dir-fwd': False,
    'dir-rev': True,
}
        
PE_DIR_SCHEMA: Dict[str, List[str]] = {
    'dir-AP': [],
    'dir-PA': [],
    'dir-fwd': [],
    'dir-rev': [],
}

class BidsReaderInput(BaseModel):
    bids_dir: Path

class BidsReader(BidsReaderInput):

    def __init__(self, bids_dir: Path):
        input_data = BidsReaderInput(bids_dir = bids_dir)
        super().__init__(**input_data.dict())

    def get_subjects(self) -> List[str]:
        subdirs = []
        for i in self.bids_dir.iterdir():
            if i.is_dir() and i.name.startswith('sub-'):
                subdirs.append(i.stem)
        subdirs.sort()

        return subdirs
    
    def get_sessions(self, sub_id: str) -> List[str]:
        subdirs = []
        sub_dir = Path(
            f"{self.bids_dir}/{self._process_dir(sub_id,'sub')}"
        )
        assert sub_dir.exists(), f"Directory [{sub_dir}] does not exist."
        
        for i in sub_dir.iterdir():
            if i.is_dir() and i.name.startswith('ses-'):
                subdirs.append(i.stem)        
        subdirs.sort()

        return subdirs
    
    def get_anat(
        self,
        sub_id: str,
        ses_id: Optional[str] = None,
        contrast_type: str = 'T2w',
    ) -> Path:
        # If multiple T2w runs detected, grab the last one
        runs = []
        if ses_id is not None:
            # Find a T2w given a subject and session ID
            sub_ses_anat_dir = Path(self.bids_dir) / self._process_dir(sub_id, 'sub') / self._process_dir(ses_id, 'ses') / 'anat'
            for i in sub_ses_anat_dir.iterdir():
                if not i.is_dir() and i.name.endswith(f'_T2w.nii.gz'):
                    runs.append(i)
        else:
            # Find any T2w across all sessions
            ses_ids = self.get_sessions(sub_id)
            for _ses_id in ses_ids:
                sub_ses_anat_dir = Path(self.bids_dir) / self._process_dir(sub_id, 'sub') / self._process_dir(_ses_id, 'ses') / 'anat'
                for i in sub_ses_anat_dir.iterdir():
                    if not i.is_dir() and i.name.endswith(f'_T2w.nii.gz'):
                        runs.append(i)
        
        n_runs = len(runs)
        assert n_runs > 0, f"Warning: {n_runs} detected."

        if n_runs > 1:
            print(f"Warning: Multiple runs detected, using {runs[-1].stem}")

        return runs[-1]

    def get_bold_runs(
        self,
        sub_id: str,
        ses_id: str,
        ignore_tasks: List[str] = []
    ) -> Dict[str, List[Path]]:
        ignore_tasks = [f"task-{task_id}" if not task_id.startswith('task-') else task_id for task_id in ignore_tasks]

        runs_dict = deepcopy(PE_DIR_SCHEMA)

        sub_ses_func_dir = Path(self.bids_dir) / self._process_dir(sub_id, 'sub') / self._process_dir(ses_id, 'ses') / 'func'
        assert sub_ses_func_dir.exists(), f"Directory [{sub_ses_func_dir}] does not exist."

        for i in sub_ses_func_dir.iterdir():
            if any(ignore_task in i.stem for ignore_task in ignore_tasks):
                continue

            if not i.is_dir() and i.name.endswith("_bold.nii.gz"):
                _dir = i.stem.split('_dir-')[-1].split('_')[0]
                _dir = f"dir-{_dir}"
                assert _dir in runs_dict.keys(), f"{_dir} is not supported."
                runs_dict[_dir].append(i)

        # sort
        for k, runs in runs_dict.items():
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
        
        """
        Currently, only supports TOPUP distortion correction
        """

        runs_dict = deepcopy(PE_DIR_SCHEMA)
        sub_ses_fmap_dir = Path(self.bids_dir) / self._process_dir(sub_id, 'sub') / self._process_dir(ses_id, 'ses') / 'fmap'
        #assert sub_ses_fmap_dir.exists(), f"Directory [{sub_ses_fmap_dir}] does not exist."

        if sub_ses_fmap_dir.exists():
            for i in sub_ses_fmap_dir.iterdir():
                if not i.is_dir() and i.name.endswith("_epi.nii.gz"):
                    _dir = i.stem.split('_dir-')[-1].split('_')[0]
                    _dir = f"dir-{_dir}"
                    assert _dir in runs_dict.keys(), f"{_dir} is not supported."
                    runs_dict[_dir].append(i)
            # sort
            for k, runs in runs_dict.items():
                runs.sort()

        return runs_dict
    
    def _remove_phase_parts(self, bold_list: List[Path]) -> List[Path]:

        return [bold_path for bold_path in bold_list if '_part-phase_' not in str(bold_path)]
    
    def _process_dir(self, entry, prefix: str) -> str:

        if not entry.startswith(f"{prefix}-"):
            entry = f"{prefix}-{entry}"

        return entry