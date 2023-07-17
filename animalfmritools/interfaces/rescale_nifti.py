from nipype.interfaces.base import (
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

import os

OUTPATH = "rescaled.nii.gz"


def _RescaleNifti(nifti_path, rescale_factor, out_path=OUTPATH):
    import numpy as np
    import nibabel as nib

    # Load nifti
    img = nib.load(nifti_path)
    header = img.header

    # create a copy not a view, so it does not change if original changed
    qformcode = header["qform_code"].copy()
    sformcode = header["sform_code"].copy()

    # now to augment the image you need to change both the sform and the qform
    # augmenting pixidm, will change the qform
    header["pixdim"][1:4] = header["pixdim"][1:4] * rescale_factor

    # augmenting the affine, will change the sform
    # !!! YOU NEED TO CHANGE THAT BASED ON YOUR ORIENTATION
    img.affine[0][0] = img.affine[0][0] * rescale_factor
    img.affine[1][1] = img.affine[1][1] * rescale_factor
    img.affine[2][2] = img.affine[2][2] * rescale_factor

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

    # you save the image with the same name to overwrite it
    nib.save(new_img_aug, out_path)


class RescaleNiftiInputSpec(TraitedSpec):
    nifti_path = File(exists=True, desc="nifti path", mandatory=True)
    rescale_factor = traits.Float(desc="rescale factor", mandatory=True)


class RescaleNiftiOutputSpec(TraitedSpec):
    rescaled_path = File(exists=True, desc="rescaled nifti path")


class RescaleNifti(SimpleInterface):
    input_spec = RescaleNiftiInputSpec
    output_spec = RescaleNiftiOutputSpec

    def _run_interface(self, runtime):
        _RescaleNifti(
            self.inputs.nifti_path,
            self.inputs.rescale_factor,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["rescaled_path"] = os.path.abspath(OUTPATH)

        return outputs
