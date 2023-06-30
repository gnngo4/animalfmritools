from nipype.interfaces.base import (
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

import os

OUTPATH = "evenified.nii.gz"
EXPECTED_N_DIMS = [3, 4]

def _EvenifyNifti(nifti_path, out_path=OUTPATH):

    import numpy as np
    import nibabel as nib

    img = nib.load(nifti_path)
    data = img.get_fdata()
    n_dims = len(data.shape)
    if n_dims not in EXPECTED_N_DIMS:
        raise ValueError("The value must be 3 or 4.")
    # Check whether the dimensions of XYZ are odd
    # if odd, truncate a slice to make it even
    for ix, dim_size in enumerate(data.shape[:3]):
        if dim_size % 2 == 1:
            if ix == 0:
                data = data[:-1,:,:] if n_dims == 3 else data[:-1,:,:,:]
            elif ix == 1:
                data = data[:,:-1,:] if n_dims == 3 else data[:,:-1,:,:]
            elif ix == 2:
                data = data[:,:,:-1] if n_dims == 3 else data[:,:,:-1,:]

    nib.save(
        nib.Nifti1Image(data, img.affine),
        out_path
    )

class EvenifyNiftiInputSpec(TraitedSpec):
    nifti_path = File(exists=True, desc="nifti path", mandatory=True)

class EvenifyNiftiOutputSpec(TraitedSpec):
    out_path = File(exists=True, desc="nifti path with even dimensions enforced")


class EvenifyNifti(SimpleInterface):
    input_spec = EvenifyNiftiInputSpec
    output_spec = EvenifyNiftiOutputSpec

    def _run_interface(self, runtime):
        _EvenifyNifti(
            self.inputs.nifti_path,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_path"] = os.path.abspath(OUTPATH)

        return outputs