import itertools
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.signal import clean
from pydantic import BaseModel, validator

sys.path.append("/opt/animalfmritools")
from notebooks.local.preprocess.susan import SUSAN
from notebooks.local.preprocess.workbench import CiftiConvertToNifti

SUPPORTED_SPECIES = ["marmoset", "mouse", "rat"]

EXPECTED_CONFOUNDS_COLUMNS = [
    "framewise_displacement",
    "global_signal",
    "global_signal_derivative1",
    "global_signal_power2",
    "global_signal_derivative1_power2",
    "csf",
    "csf_derivative1",
    "csf_power2",
    "csf_derivative1_power2",
    "white_matter",
    "white_matter_derivative1",
    "white_matter_power2",
    "white_matter_derivative1_power2",
    "trans_x",
    "trans_x_derivative1",
    "trans_x_derivative1_power2",
    "trans_x_power2",
    "trans_y",
    "trans_y_derivative1",
    "trans_y_power2",
    "trans_y_derivative1_power2",
    "trans_z",
    "trans_z_derivative1",
    "trans_z_power2",
    "trans_z_derivative1_power2",
    "rot_x",
    "rot_x_derivative1",
    "rot_x_derivative1_power2",
    "rot_x_power2",
    "rot_y",
    "rot_y_derivative1",
    "rot_y_power2",
    "rot_y_derivative1_power2",
    "rot_z",
    "rot_z_derivative1",
    "rot_z_derivative1_power2",
    "rot_z_power2",
]


class ConfoundsTSVInput(BaseModel):
    confounds_tsv: Path

    @validator('confounds_tsv')
    def validate_tsv_headers(cls, value):
        try:
            cols = pd.read_csv(value, sep='\t').columns
            for expected_col in EXPECTED_CONFOUNDS_COLUMNS:
                if expected_col not in cols:
                    raise ValueError(f"Expected column [{expected_col}] not found in {value}.")
        except FileNotFoundError as err:
            raise ValueError(f"{value} not found.") from err
        except StopIteration:
            raise ValueError(f"{value} is empty.") from None

        return value


class ConfoundsTSV(ConfoundsTSVInput):
    def __init__(self, confounds_tsv: Path) -> None:
        input_data = ConfoundsTSVInput(confounds_tsv=confounds_tsv)
        super().__init__(**input_data.dict())

    def load_regressors(self, rules: Tuple[str]) -> pd.DataFrame:
        confounds_df = self._load_regressors()
        confound_cols = self._get_all_regressor_labels(rules)

        return confounds_df[confound_cols]

    def _load_regressors(self) -> pd.DataFrame:
        return pd.read_csv(self.confounds_tsv, sep='\t')

    def _get_all_regressor_labels(self, rules: Optional[Tuple[str]] = None) -> List[str]:
        regressor_labels = []
        if rules is None:
            return []

        for regressor_rule in rules:
            regressor_labels += self._get_regressor_labels(regressor_rule)
        regressor_labels = list(set(regressor_labels))
        for regressor_label in regressor_labels:
            if regressor_label not in EXPECTED_CONFOUNDS_COLUMNS:
                raise ValueError(f"Regressor label [{regressor_label} not supported.")
        regressor_labels.sort()

        return regressor_labels

    def _get_regressor_labels(self, regressor_rule: str) -> List[str]:
        DERIVATIVE_SUFFICES = ["", "_derivative1", "_power2", "_derivative1_power2"]

        if regressor_rule == "mc6":
            return [f"{j}_{i}" for i, j in itertools.product(["x", "y", "z"], ["trans", "rot"])]
        elif regressor_rule == "mc24":
            return [
                f"{i}{j}"
                for i, j in itertools.product(
                    self._get_regressor_labels("mc6"),
                    DERIVATIVE_SUFFICES,
                )
            ]
        elif regressor_rule == "global":
            return ["global_signal"]
        elif regressor_rule == "global+derivatives":
            return [
                f"{i}{j}"
                for i, j in itertools.product(
                    self._get_regressor_labels("global"),
                    DERIVATIVE_SUFFICES,
                )
            ]
        elif regressor_rule == "wm":
            return ["white_matter"]
        elif regressor_rule == "wm+derivatives":
            return [
                f"{i}{j}"
                for i, j in itertools.product(
                    self._get_regressor_labels("wm"),
                    DERIVATIVE_SUFFICES,
                )
            ]
        elif regressor_rule == "csf":
            return ["csf"]
        elif regressor_rule == "csf+derivatives":
            return [
                f"{i}{j}"
                for i, j in itertools.product(
                    self._get_regressor_labels("csf"),
                    DERIVATIVE_SUFFICES,
                )
            ]
        elif regressor_rule == "motion_summary_metrics":
            return ["framewise_displacement"]
        else:
            raise ValueError(f"{regressor_rule} is not supported.")


class PreprocBoldRunInput(BaseModel):
    run_id: str
    run_metadata: Optional[Path]
    template_nifti: Path
    template_cifti: Path
    confounds_tsv: Path
    confounds_timeseries: Path
    repetition_time: Optional[float]

    @validator('template_nifti', 'template_cifti', 'confounds_tsv', 'confounds_timeseries')
    def check_path_exists(cls, v):
        if not v.exists():
            raise ValueError(f"{v} does not exist.")
        return v

    @validator('repetition_time', always=True)
    def extract_repetition_time(cls, v, values):
        if v is not None:
            # repetition_time was manually set.
            return v
        if values.get('run_metadata') is not None:
            # extract repetition_time from metadata
            import json

            with open(values['run_metadata'], "r") as f:
                data = json.load(f)
            return data['RepetitionTime']
        if values['run_metadata'] is None:
            raise ValueError("No metadata found. Set `repetition_time` manually.")
        else:
            raise ValueError(
                f"Unable to extract repetition time from {values['run_metadata']}. Set `repetition_time` manually."
            )


class PreprocBoldRun(PreprocBoldRunInput):
    def __init__(
        self,
        run_id: str,
        template_nifti: Path,
        template_cifti: Path,
        confounds_tsv: Path,
        confounds_timeseries: Path,
        run_metadata: Optional[Path] = None,
        repetition_time: Optional[float] = None,
    ) -> None:
        input_data = PreprocBoldRunInput(
            run_id=run_id,
            run_metadata=run_metadata,
            template_nifti=template_nifti,
            template_cifti=template_cifti,
            confounds_tsv=confounds_tsv,
            confounds_timeseries=confounds_timeseries,
            repetition_time=repetition_time,
        )
        super().__init__(**input_data.dict())

    def load_confounds(self, regressor_rules: List[str]) -> pd.DataFrame:
        tsv = ConfoundsTSV(self.confounds_tsv)
        return tsv.load_regressors(regressor_rules)

    @lru_cache(maxsize=4)
    def load_bold(
        self,
        data_type: str,
        regressor_rules: Optional[Tuple[str]] = None,
        n_start_volumes_to_remove: int = 1,
        detrend: bool = True,
        standardize: str = "zscore_sample",
        standardize_confounds: bool = True,
        filter: str = "butterworth",
        low_pass: float = 0.2,
        high_pass: float = 0.01,
        smooth_mm: Optional[float] = None,
        out_dir: Optional[Path] = None,
        pseudo_nifti: str = "/tmp/nifti_placeholder.nii.gz",
        tmp_denoised_path="/tmp/denoised.nii.gz",
    ) -> Dict[str, np.ndarray]:
        """
        Loads and preprocessed BOLD data

        Args:
            data_type: The type of input data ('nifti' or 'cifti').
            regressor_rules: A tuple of strings defining rules for selecting confounds
                                from the confound file.
                                Supported rules:
                                    - 'mc{6,24}': 6 or 24 motion parameters.
                                    - 'wm_csf_global': White matter, cerebrospinal fluid, and global signal.
                                    - 'wm_csf_global_derivatives': As above, plus their derivatives.
                                    - 'motion_summary_metrics': Motion summary metrics.
                                Defaults to None.
            n_start_volumes_to_remove: Number of initial volumes to discard. Defaults to 1.
            detrend: Whether to detrend the BOLD data. Defaults to True.
            standardize: Standardization method (e.g., 'zscore_sample'). Defaults to 'zscore_sample'.
            standardize_confounds: Whether to standardize confounds. Defaults to True.
            filter: Type of temporal filter to apply (e.g., 'butterworth'). Defaults to 'butterworth'.
            low_pass: High-pass filter cutoff frequency in Hz. Defaults to 0.2.
            high_pass: Low-pass filter cutoff frequency in Hz. Defaults to 0.01.
            smooth_mm: Smoothing parameter in mm. Defaults to None.
            out_dir: Output directory. It will output smoothed NIFTI images to this location. Defaults to None.
            pseudo_nifti: Path to a temporary NIfTI file used for CIFTI data.
            tmp_denoised_path: Path to a temporary NIfTI file prior to running smoothing.

        Returns:
            A dictionary containing:
                - 'raw': The raw BOLD data.
                - 'denoised': The preprocessed (denoised) BOLD data.
        """

        if smooth_mm is None:
            assert out_dir is not None, "Must set `out_dir`."
            assert out_dir.exists(), f"{out_dir} does not exist."

        if data_type not in ["cifti", "nifti"]:
            raise ValueError("data_type must be set to cifti or nifti.")

        # Get confounds
        confounds = self.load_confounds(regressor_rules)

        # Load self.template_cifti
        if data_type == "nifti":
            print(f"Loading: {self.template_nifti}")
            bold_img = nib.load(self.template_nifti)
            assert len(bold_img.shape) == 4, f"bold_img [{bold_img.shape}] is expected to be 4d."
            # Remove starting volumes from bold data and regressors
            bold_img = bold_img.slicer[:, :, :, n_start_volumes_to_remove:]
            bold_data = bold_img.get_fdata()
            if regressor_rules is None or len(regressor_rules) == 0:
                regressors = None
            else:
                regressors = confounds.values[n_start_volumes_to_remove:, :]
            # Denoise with nuisance regressors
            (x, y, z, t) = bold_data.shape
            bold_data_reshape = bold_data.reshape(-1, t).T
            denoised_bold_data = clean(
                bold_data_reshape,
                detrend=detrend,
                standardize=standardize,
                confounds=regressors,
                standardize_confounds=standardize_confounds,
                filter=filter,
                low_pass=low_pass,
                high_pass=high_pass,
                t_r=self.repetition_time,
            )
            denoised_bold_data = denoised_bold_data.T.reshape(x, y, z, t)
            # Check raw and denoised data matches dimensions
            assert np.all(bold_data_reshape.T.reshape(x, y, z, t) == bold_data)
            assert bold_data.shape == denoised_bold_data.shape

            # Note: nilearn.signal.clean() inputs (n_samples, n_features) or (n_timepoints, n_voxels)
            # Pseudo-nifti is loaded as (n_voxels, n_timepoints) -> this is transposed to be used in nilearn.signal.clean()
            # After denoising, we transpose back to retain bold timeseries as (n_voxels, n_timepoints)
            if smooth_mm is None:
                return {
                    "raw": bold_data,
                    "denoised": denoised_bold_data,
                }
            else:
                # Skip if already exists
                smooth_mm_str = str(smooth_mm).replace(".", "-")
                smoothed_denoised_data = str(self.template_nifti).replace(
                    "bold.nii.gz", f"desc-denoised_s-{smooth_mm_str}_bold.nii.gz"
                )
                smoothed_denoised_data = out_dir / smoothed_denoised_data.split("/")[-1]
                if smoothed_denoised_data.exists():
                    print(f"[nifti] Skip smoothing: {smooth_mm} mm. Path exists: {Path(smoothed_denoised_data)}")
                    return {
                        "raw": bold_data,
                        "denoised": nib.load(smoothed_denoised_data),
                    }
                print(f"[nifti] Smoothing: {smooth_mm} mm. Generating: {smoothed_denoised_data}")
                # Smooth data
                bold_mean = bold_data.mean(-1)
                img = nib.load(self.template_nifti)
                # Add mean back to denoised data
                denoised_data = denoised_bold_data + bold_mean[:, :, :, np.newaxis]
                denoised_img = nib.Nifti1Image(denoised_data, affine=img.affine, header=img.header)
                nib.save(denoised_img, tmp_denoised_path)
                susan = SUSAN(tmp_denoised_path, smoothed_denoised_data, smooth_mm)
                _ = susan.execute()
                return {
                    "raw": bold_data,
                    "denoised": nib.load(smoothed_denoised_data),
                }

        elif data_type == "cifti":
            # Convert cifti to pseudo-nifti
            cifti_to_nifti = CiftiConvertToNifti(str(self.template_cifti), pseudo_nifti)
            _ = cifti_to_nifti.execute()
            # Load pseudo-nifti
            bold_img = nib.load(pseudo_nifti)
            assert len(bold_img.shape) == 4, f"bold_img [{bold_img.shape}] is expected to be 4d."
            assert bold_img.shape[1:3] == (
                1,
                1,
            ), f"bold_img [{bold_img.shape}] 2nd and 3rd dimension is expected to be 1."
            # Remove starting volumes from bold data and regressors
            bold_img = bold_img.slicer[:, :, :, n_start_volumes_to_remove:]
            bold_data = bold_img.get_fdata()
            bold_data = bold_data[:, 0, 0, :].T
            if regressor_rules is None or len(regressor_rules) == 0:
                regressors = None
            else:
                regressors = confounds.values[n_start_volumes_to_remove:, :]
            # Delete pseudo-nifti
            cifti_to_nifti.delete_outputs()
            # Denoise with nuisance regressors
            denoised_bold_data = clean(
                bold_data,
                detrend=detrend,
                standardize=standardize,
                confounds=regressors,
                standardize_confounds=standardize_confounds,
                filter=filter,
                low_pass=low_pass,
                high_pass=high_pass,
                t_r=self.repetition_time,
            )
            # Check raw and denoised data matches dimensions
            assert bold_data.shape == denoised_bold_data.shape

            # Note: nilearn.signal.clean() inputs (n_samples, n_features) or (n_timepoints, n_voxels)
            # Pseudo-nifti is loaded as (n_voxels, n_timepoints) -> this is transposed to be used in nilearn.signal.clean()
            # After denoising, we transpose back to retain bold timeseries as (n_voxels, n_timepoints)
            return {
                "raw": bold_data.T,
                "denoised": denoised_bold_data.T,
            }

    def load_correlation_matrix(
        self,
        data_type: str,
        denoise: bool,
        pseudo_nifti: str = "/tmp/nifti_placeholder.nii.gz",
        regressor_rules: Optional[Tuple[str]] = None,
        n_start_volumes_to_remove: int = 1,
        detrend: bool = True,
        standardize: str = "zscore_sample",
        standardize_confounds: bool = True,
        filter: str = "butterworth",
        low_pass: float = 0.2,
        high_pass: float = 0.01,
    ) -> np.ndarray:
        if data_type not in ["cifti", "nifti"]:
            raise ValueError("data_type must be set to cifti or nifti.")

        parameters = {
            "data_type": data_type,
            "regressor_rules": regressor_rules,
            "n_start_volumes_to_remove": n_start_volumes_to_remove,
            "detrend": detrend,
            "standardize": standardize,
            "standardize_confounds": standardize_confounds,
            "filter": filter,
            "low_pass": low_pass,
            "high_pass": high_pass,
        }

        bold_data = self.load_bold(**parameters)

        if denoise:
            return np.corrcoef(self._zscore_normalize(bold_data["denoised"]))
        else:
            return np.corrcoef(self._zscore_normalize(bold_data["raw"]))

    def _zscore_normalize(self, _bold_data: np.ndarray) -> np.ndarray:
        return (_bold_data - _bold_data.mean(1, keepdims=True)) / _bold_data.std(1, keepdims=True)

    def __hash__(self):
        # Since PreprocBoldRunInput is immutable and hashable, you can hash its attributes directly
        return hash(
            (
                self.run_id,
                self.template_nifti,
                self.template_cifti,
                self.confounds_tsv,
                self.confounds_timeseries,
                self.run_metadata,
                self.repetition_time,
            )
        )

    def __eq__(self, other):
        # Implement equality comparison based on the attributes that determine the object's identity
        return isinstance(other, PreprocBoldRun) and self.__dict__ == other.__dict__


class Session(BaseModel):
    session_id: str
    bold_runs: List[PreprocBoldRun]
    repetition_time: Optional[float] = None


class Subject(BaseModel):
    subject_id: str
    sessions: List[Session]
    repetition_time: Optional[float] = None


class Dataset(BaseModel):
    dataset_id: str
    species_id: str
    subjects: List[Subject]

    @validator('species_id')
    def check_species_is_supported(cls, v):
        if v not in SUPPORTED_SPECIES:
            raise ValueError(f"{v} not in {SUPPORTED_SPECIES}.")
        return v


class PipelineReaderInput(BaseModel):
    """Input model for animalfmritools minimal preprocessing pipeline"""

    bids_dir: Path
    deriv_dir: str
    default_repetition_time: Optional[float] = (None,)
    session_tr_changes: Dict[Tuple[str, str], float] = ({},)

    @validator('bids_dir')
    def check_bids_dir_exists(cls, v):
        if not v.exists():
            raise ValueError(f"{v} does not exist.")
        return v

    @validator('deriv_dir')
    def check_deriv_dir_exists(cls, v, values):
        bids_dir = values.get('bids_dir')
        if bids_dir is None:
            raise ValueError(f"{bids_dir} was invalidated.")
        deriv_path = _get_derivatives_dir(bids_dir, v)
        if not deriv_path.exists():
            raise ValueError(f"{deriv_path} does not exist.")
        return v


class PipelineReader(PipelineReaderInput):
    """Animalfmritools minimal preprocessing pipeline reader."""

    def __init__(
        self,
        bids_dir: Path,
        deriv_dir: str,
        default_repetition_time: Optional[float] = None,
        session_tr_changes: Dict[Tuple[str, str], float] = {},
    ):
        input_data = PipelineReaderInput(
            bids_dir=bids_dir,
            deriv_dir=deriv_dir,
            default_repetition_time=default_repetition_time,
            session_tr_changes=session_tr_changes,
        )
        super().__init__(**input_data.dict())

        self.deriv_dir: Path = _get_derivatives_dir(self.bids_dir, self.deriv_dir)
        self.default_repetition_time = default_repetition_time
        self.session_tr_changes = session_tr_changes
        self._check_for_missing_subjects_in_derivative_directory()

    def _check_for_missing_subjects_in_derivative_directory(self) -> None:
        bids_sub_ids = self._find_subdirectories(self.bids_dir, stem_only=True)
        deriv_sub_ids = self._find_subdirectories(self.deriv_dir, stem_only=True)
        assert len(deriv_sub_ids) <= len(bids_sub_ids)

        bids_sub_ids = set(bids_sub_ids)
        deriv_sub_ids = set(deriv_sub_ids)
        unprocessed_sub_ids = bids_sub_ids - deriv_sub_ids
        if len(unprocessed_sub_ids) > 0:
            for sub_id in unprocessed_sub_ids:
                print(f"Warning: {sub_id} not processed.")
        else:
            print("[PipelineReader] All subjects are processed.")

    def get_processed_data(self, species_id: Optional[str] = None) -> Dataset:
        if species_id is None:
            species_id = self.bids_dir.parent.stem

        dataset = Dataset(
            dataset_id=self.bids_dir.stem,
            species_id=species_id,
            subjects=self._get_subjects(),
        )

        return dataset

    def _get_subjects(self) -> List[Subject]:
        sub_ids = self._find_subdirectories(self.deriv_dir, stem_only=True)
        assert len(sub_ids) > 0, f"No subjects found in {self.deriv_dir}."
        return [
            Subject(
                subject_id=sub_id,
                sessions=self._get_sessions(sub_id),
            )
            for sub_id in sub_ids
        ]

    def _get_sessions(self, sub_id: str) -> List[Session]:
        sub_dir = self.deriv_dir / sub_id
        ses_ids = self._find_subdirectories(sub_dir, startswith_str="ses-", stem_only=True)
        assert len(ses_ids) > 0, f"No sessions found in {sub_dir}."
        return [Session(session_id=ses_id, bold_runs=self._get_preproc_bold_runs(sub_id, ses_id)) for ses_id in ses_ids]

    def _get_preproc_bold_runs(self, sub_id: str, ses_id: str) -> List[PreprocBoldRun]:
        sub_ses_func_dir = self.deriv_dir / sub_id / ses_id / "func"
        assert sub_ses_func_dir.exists(), f"{sub_ses_func_dir} does not exist."

        preproc_bold_runs = []
        for preproc_bold in sub_ses_func_dir.glob("*_space-template_desc-preproc_bold.nii.gz"):
            preproc_bold_runs.append(self._get_preproc_bold_run(preproc_bold))

        return preproc_bold_runs

    def _get_preproc_bold_run(self, preproc_bold: Path) -> PreprocBoldRun:
        # get key-pairs of `preproc_bold`
        bids_key_pairs = self._get_all_bids_key_pairs_from_bids_string(preproc_bold.stem)
        # get metadata of the bold run
        run_metadata = self._get_run_metadata(Path(str(preproc_bold).split("derivatives")[0]), bids_key_pairs)
        # set TR
        if self.default_repetition_time is None:
            repetition_time = None
        else:
            repetition_time = self.default_repetition_time
        #
        for (sub_id, ses_id), _repetition_time in self.session_tr_changes.items():
            if bids_key_pairs['sub'] == sub_id and bids_key_pairs["ses"] == ses_id:
                repetition_time = _repetition_time

        return PreprocBoldRun(
            run_id=self._get_bids_key_from_bids_string(preproc_bold.stem, "run"),
            template_nifti=preproc_bold,
            template_cifti=Path(str(preproc_bold).replace(".nii.gz", ".dtseries.nii")),
            confounds_tsv=Path(
                str(preproc_bold).replace("space-template_desc-preproc_bold.nii.gz", "desc-confounds_timeseries.tsv")
            ),
            confounds_timeseries=Path(
                str(preproc_bold).replace("space-template_desc-preproc_bold.nii.gz", "desc-confounds_timeseries.json")
            ),
            run_metadata=run_metadata,
            repetition_time=repetition_time,
        )

    def _get_bids_key_from_bids_string(self, bids_string: str, bids_key: str) -> str:
        if f"{bids_key}-" not in bids_string:
            raise ValueError(f"{bids_key}_id not found in {bids_string}")
        _id = bids_string.split(f'{bids_key}-')[1].split('_')[0]

        return f"{bids_key}-{_id}"

    def _get_all_bids_key_pairs_from_bids_string(self, bids_string: str) -> Dict[str, str]:
        BIDS_KEYS = ["sub", "ses", "task", "dir", "run"]
        bids_key_pairs = {}
        for bids_key in BIDS_KEYS:
            bids_key_pairs[bids_key] = self._get_bids_key_from_bids_string(bids_string, bids_key)

        return bids_key_pairs

    def _get_run_metadata(self, bids_dir: Path, bids_key_pairs: Dict[str, str]) -> Optional[Path]:
        sub_id = bids_key_pairs["sub"]
        ses_id = bids_key_pairs["ses"]
        task_id = bids_key_pairs["task"]
        dir_id = bids_key_pairs["dir"]
        run_id = bids_key_pairs["run"]
        json_stem = f"{sub_id}_{ses_id}_{task_id}_{dir_id}_{run_id}_bold.json"
        metadata_json = bids_dir / sub_id / ses_id / "func" / json_stem
        if not metadata_json.exists():
            return None
        else:
            return metadata_json

    def _find_subdirectories(self, input_dir: Path, startswith_str: str = "sub-", stem_only: bool = False) -> List[str]:
        """Find all subdirectories within the given input directory that start with "sub-"."""
        input_dir = Path(input_dir)
        subdirectories = []
        for subdir in input_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith(startswith_str):
                if stem_only:
                    subdirectories.append(subdir.stem)
                else:
                    subdirectories.append(subdir)
        return subdirectories


def _get_derivatives_dir(bids_dir: Path, deriv_dir: str) -> Path:
    return bids_dir / "derivatives" / deriv_dir
