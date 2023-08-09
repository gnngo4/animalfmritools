import os

from nipype.interfaces.base import (
    File,
    SimpleInterface,
    TraitedSpec,
)

OUTPATH = "copied.nii.gz"


def _CopyAffineHeaderInfo(input_image, reference_image, out_path=OUTPATH):
    import nibabel as nib

    input_img = nib.load(input_image)
    ref_img = nib.load(reference_image)

    nib.save(
        nib.Nifti1Image(input_img.get_fdata(), header=ref_img.header, affine=ref_img.affine),
        out_path,
    )


class CopyAffineHeaderInfoInputSpec(TraitedSpec):
    input_image = File(exists=True, desc="input_image", mandatory=True)
    reference_image = File(exists=True, desc="reference_image", mandatory=True)


class CopyAffineHeaderInfoOutputSpec(TraitedSpec):
    copied_path = File(exists=True, desc="copied nifti path")


class CopyAffineHeaderInfo(SimpleInterface):
    input_spec = CopyAffineHeaderInfoInputSpec
    output_spec = CopyAffineHeaderInfoOutputSpec

    def _run_interface(self, runtime):
        _CopyAffineHeaderInfo(
            self.inputs.input_image,
            self.inputs.reference_image,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["copied_path"] = os.path.abspath(OUTPATH)

        return outputs
