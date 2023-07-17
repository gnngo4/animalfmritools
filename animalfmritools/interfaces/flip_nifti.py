from nipype.interfaces.base import (
    File,
    SimpleInterface,
    TraitedSpec,
)

import os

OUTPATH = "flipped.nii.gz"


def _FlipNifti(nifti_path, out_path=OUTPATH):
    import numpy as np
    import nibabel as nib

    img = nib.load(nifti_path)
    data = img.get_fdata()
    # Flip all XYZ dimensions
    for i in range(3):
        data = np.flip(data, axis=i)

    nib.save(nib.Nifti1Image(data, img.affine), out_path)


class FlipNiftiInputSpec(TraitedSpec):
    nifti_path = File(exists=True, desc="nifti path", mandatory=True)


class FlipNiftiOutputSpec(TraitedSpec):
    flipped_path = File(exists=True, desc="flipped nifti path")


class FlipNifti(SimpleInterface):
    input_spec = FlipNiftiInputSpec
    output_spec = FlipNiftiOutputSpec

    def _run_interface(self, runtime):
        _FlipNifti(
            self.inputs.nifti_path,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["flipped_path"] = os.path.abspath(OUTPATH)

        return outputs
