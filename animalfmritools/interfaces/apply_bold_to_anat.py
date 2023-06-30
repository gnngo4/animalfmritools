from nipype.interfaces.base import (
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

import os

BOLD_TO_ANAT = "space-t2_bold.nii.gz"


def _ApplyBoldToAnat(
    bold_path,
    hmc_mats,
    bold_to_anat_warp,
    anat_resampled,
    debug,
):
    from nipype.interfaces import fsl
    import nibabel as nib

    split_bold = fsl.Split(dimension="t", in_file=bold_path)
    res = split_bold.run()
    bold_list = res.outputs.out_files

    vol_t1_bold = []
    assert len(bold_list) == len(
        hmc_mats
    ), "hmc mats and split bold data are not equal lengths."
    for ix, (vol_mat, vol_bold) in enumerate(zip(hmc_mats, bold_list)):
        if debug and ix == 10:
            break

        # Combine `vol_mat` with `bold_to_anat_warp``
        convert_warp = fsl.ConvertWarp(
            reference=anat_resampled,
            premat=vol_mat,
            warp1=bold_to_anat_warp,
        )
        res = convert_warp.run()
        vol_warp = res.outputs.out_file
        # Apply the new warp to `vol_bold`
        apply_warp = fsl.ApplyWarp(
            in_file=vol_bold,
            ref_file=anat_resampled,
            field_file=vol_warp,
        )
        res = apply_warp.run()
        vol_out = res.outputs.out_file

        vol_t1_bold.append(vol_out)

        if ix == 0:
            print(
                f"""
            Command examples for one iteration of merging
            hmc affine and t1 warp and applying the warp.
            [cmd] Merge affine and warp:
            {convert_warp.cmdline}
            [cmd] Apply merged warp:
            {apply_warp.cmdline}
            """
            )

    # Verbose
    print("Merge the following volumes:")
    for _vol in vol_t1_bold:
        print(f"    - {_vol}")
    # Save merged volume as nifti
    merged_nii = nib.funcs.concat_images(vol_t1_bold)
    nib.save(merged_nii, BOLD_TO_ANAT)
    # Assertion
    assert os.path.exists(BOLD_TO_ANAT), f"{BOLD_TO_ANAT} was not created."


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
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["t1_bold_path"] = os.path.abspath(BOLD_TO_ANAT)

        return outputs