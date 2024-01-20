import os

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

OUTPATH = "rescaled.nii.gz"


def _RescaleNifti(nifti_path, rescale_factor, force_translation=None, out_path=OUTPATH):
    # Load nifti
    img = nib.load(nifti_path)
    header = img.header.copy()

    # Create copy of qform and sform codes
    qformcode = header["qform_code"].copy()
    sformcode = header["sform_code"].copy()

    # Apply a rescale factor to pixel dimensions
    header["pixdim"][1:4] *= rescale_factor

    # Apply rescale factor to affine matrix diagonal elements
    img.affine[0][0] *= rescale_factor
    img.affine[1][1] *= rescale_factor
    img.affine[2][2] *= rescale_factor

    # set q and sformcode (does not help) that's why I reimport
    header["qform_code"] = qformcode
    header["sform_code"] = sformcode

    # now, you create a new image using three pieces of info:
    # matrix values, affine matrix, header
    values = np.array(img.get_fdata())
    new_img = nib.Nifti1Image(values, img.affine, header)

    # save the augmented image
    nib.save(new_img, out_path)

    # reimport the image, change the sformcode and qformcode because they change if you set them in last step
    new_img_abs = os.path.abspath(out_path)

    img_aug = nib.load(new_img_abs)
    header_aug = img_aug.header

    header_aug["qform_code"] = qformcode
    header_aug["sform_code"] = sformcode

    values_aug = np.array(img_aug.get_fdata())
    new_img_aug = nib.Nifti1Image(values_aug, img_aug.affine, header_aug)

    if force_translation is not None:
        new_img_aug.affine[:3, 3] = force_translation

    # you save the image with the same name to overwrite it
    nib.save(new_img_aug, out_path)


class RescaleNiftiInputSpec(TraitedSpec):
    nifti_path = File(exists=True, desc="nifti path", mandatory=True)
    rescale_factor = traits.Float(desc="rescale factor", mandatory=True)
    force_translation = traits.Any(None, usedefault=True, desc="Update affine with a translation component")


class RescaleNiftiOutputSpec(TraitedSpec):
    rescaled_path = File(exists=True, desc="rescaled nifti path")


class RescaleNifti(SimpleInterface):
    input_spec = RescaleNiftiInputSpec
    output_spec = RescaleNiftiOutputSpec

    def _run_interface(self, runtime):
        _RescaleNifti(
            self.inputs.nifti_path,
            self.inputs.rescale_factor,
            self.inputs.force_translation,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["rescaled_path"] = os.path.abspath(OUTPATH)

        return outputs
