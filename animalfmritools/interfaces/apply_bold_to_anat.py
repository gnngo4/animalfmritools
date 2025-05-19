import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from nipype.interfaces.base import (
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

BOLD_TO_ANAT = "space-t2_bold.nii.gz"

def _process_volume(hmc_mat, vol_bold, anat_resampled, bold_to_anat_warp, vol_idx):

    from nipype.interfaces import fsl

    # Combine matrices
    convert_warp = fsl.ConvertWarp(
        reference=anat_resampled,
        premat=hmc_mat,
        warp1=bold_to_anat_warp,
        relwarp=True,
        out_file=f"vol_warp_{vol_idx}.nii.gz"
    )
    res_warp = convert_warp.run()
    vol_warp = res_warp.outputs.out_file

    # Apply warp
    apply_warp = fsl.ApplyWarp(
        in_file=vol_bold,
        ref_file=anat_resampled,
        field_file=vol_warp,
        out_file=f"vol_warped_{vol_idx}.nii.gz",
    )
    res_apply = apply_warp.run()
    vol_out = res_apply.outputs.out_file

    if os.path.exists(vol_warp):
        os.remove(vol_warp)

    return vol_out, vol_idx, convert_warp.cmdline if vol_idx == 0 else None, apply_warp.cmdline if vol_idx ==0 else None



def _ApplyBoldToAnat(
    bold_path,
    hmc_mats,
    bold_to_anat_warp,
    anat_resampled,
    debug,
    num_procs,
):
    
    import nibabel as nib
    import tempfile
    import shutil
    import gc

    # Default number of processes: total CPUs / 2
    if num_procs is None:
        import multiprocessing
        num_procs = max(1, multiprocessing.cpu_count() // 2)

    # Create tmp directory 
    tmp_dir = tempfile.mkdtemp()
    orig_dir = os.getcwd()
    os.chdir(tmp_dir)

    try:
        # Split BOLD volume efficiently
        bold_img = nib.load(bold_path)
        bold_data = bold_img.get_fdata()
        assert len(bold_data.shape) == 4, f"{bold_path} must be 4D."
        n_vols = bold_data.shape[3]

        # Debug - only run 10 volumes
        if debug:
            n_vols = min(10, n_vols)
            hmc_mats = hmc_mats[:n_vols]

        assert n_vols == len(hmc_mats), f"HMC mats [{len(hmc_mats)}] and BOLD volmes [{n_vols}] are not consistent."

        # Split volumes and save to tmp files
        bold_list = []
        for i in range(n_vols):
            vol_bold = bold_data[..., i].copy()
            vol_img = nib.Nifti1Image(vol_bold, bold_img.affine, bold_img.header)
            vol_out = f"vol_{str(i).zfill(4)}.nii.gz"
            nib.save(vol_img, vol_out)
            bold_list.append(vol_out)

        del vol_bold, vol_img
        gc.collect()

        # Create parallel preproc. list
        vol_data_list = [
            (
                hmc_mats[i],
                bold_list[i],
                anat_resampled,
                bold_to_anat_warp,
                i,
            ) for i in range(n_vols)
        ]

        # Process vols in parallel
        vol_t1_bold = [None] * n_vols # Instantiate results
        example_commands = {"convert_warp": None, "apply_warp": None}

        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            future_to_idx = {executor.submit(_process_volume, *vol_data): vol_data[4] for vol_data in vol_data_list}

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    vol_out, vol_idx, convert_cmd, apply_cmd = future.result()
                    vol_t1_bold[vol_idx] = vol_out
                    
                    # Store example commands
                    if convert_cmd:
                        example_commands["convert_warp"] = convert_cmd
                    if apply_cmd:
                        example_commands["apply_warp"] = apply_cmd
                        
                except Exception as e:
                    print(f"Error processing volume {idx}: {e}")
                    raise
        
        # Display example commands
        if example_commands["convert_warp"] and example_commands["apply_warp"]:
            print(
                f"""
                Command examples for one iteration of merging
                hmc affine and t1 warp and applying the warp.
                [cmd] Merge affine and warp:
                {example_commands["convert_warp"]}
                [cmd] Apply merged warp:
                {example_commands["apply_warp"]}
                """
            )
            
        # Ensure all volumes were processed
        assert None not in vol_t1_bold, "Not all volumes were processed."
        
        # Merge volumes efficiently
        print(f"Merging {len(vol_t1_bold)} volumes into {BOLD_TO_ANAT}")
        
        # Load all warped volumes into a 4D array
        first_img = nib.load(vol_t1_bold[0])
        shape_3d = first_img.shape
        merged_data = np.zeros(shape_3d + (n_vols,), dtype=np.float32)
        
        for i, vol_file in enumerate(vol_t1_bold):
            vol_img = nib.load(vol_file)
            merged_data[..., i] = vol_img.get_fdata()
            
        # Create and save the merged image
        merged_img = nib.Nifti1Image(merged_data, first_img.affine, first_img.header)
        output_path = os.path.join(orig_dir, BOLD_TO_ANAT)
        nib.save(merged_img, output_path)
        
        # Verify output
        assert os.path.exists(output_path), f"{output_path} was not created"
        
        return output_path
        
    finally:
        # Clean up temp directory
        os.chdir(orig_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

class ApplyBoldToAnatInputSpec(TraitedSpec):
    bold_path = File(exists=True, desc="bold path", mandatory=True)
    hmc_mats = InputMultiObject(
        File(exists=True),
        desc="list of hmc affine mat files",
        mandatory=True,
    )
    bold_to_anat_warp = File(exists=True, desc="bold to anat warp", mandatory=True)
    anat_resampled = File(
        exists=True,
        desc="t1 resampled to resolution of bold data",
        mandatory=True,
    )
    debug = traits.Bool(
        desc="debug generates bold images with the first 10 volumes",
        mandatory=True,
    )
    num_procs = traits.Int(
        desc="number of parallel processes to use (default: CPU count / 2)",
        mandatory=False,
    )


class ApplyBoldToAnatOutputSpec(TraitedSpec):
    t1_bold_path = File(exists=True, desc="transformed-to-t1 bold path")


class ApplyBoldToAnat(SimpleInterface):
    input_spec = ApplyBoldToAnatInputSpec
    output_spec = ApplyBoldToAnatOutputSpec

    def _run_interface(self, runtime):
        _ApplyBoldToAnat(
            self.inputs.bold_path,
            self.inputs.hmc_mats,
            self.inputs.bold_to_anat_warp,
            self.inputs.anat_resampled,
            self.inputs.debug,
            self.inputs.num_procs,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["t1_bold_path"] = os.path.abspath(BOLD_TO_ANAT)

        return outputs
