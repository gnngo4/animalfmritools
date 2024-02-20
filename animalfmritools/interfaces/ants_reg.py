import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    Str,
    TraitedSpec,
    traits,
)


class RegistrationSyNInputSpec(CommandLineInputSpec):
    dimension = traits.Enum(3, 2, argstr="-d %d", desc="Image dimension (2 or 3)", usedefault=True)
    fixed_image = File(argstr="-f %s", desc="Fixed (or reference image) (.nii.gz)", mandatory=True, exists=True)
    moving_image = File(argstr="-m %s", desc="Moving image (.nii.gz)", mandatory=True, exists=True)
    output_prefix = Str("transform", argstr="-o %s", desc="Prefix of outputs", usedefault=True)
    # Note: Remove option to do only deformation (i.e., -t so/bo)
    # This has implications in the output logic of method `_list_outputs`, which does not generate an `out_matrix`, and currently not supported.
    transform_type = traits.Enum(
        "s", "t", "r", "a", "sr", "b", "br", argstr="-t %s", desc="Transform type", usedefault=True
    )


class RegistrationSyNOutputSpec(TraitedSpec):
    warped_image = File(exists=True, desc="Warped image")
    inverse_warped_image = File(exists=True, desc="Inverse warped image")
    out_matrix = File(exists=True, desc="Affine matrix")
    forward_warp_field = File(exists=True, desc="Forward warp field")
    inverse_warp_field = File(exists=True, desc="Inverse warp field")


class RegistrationSyN(CommandLine):
    _cmd = "antsRegistrationSyN.sh"
    input_spec = RegistrationSyNInputSpec
    output_spec = RegistrationSyNOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_base = os.path.abspath(self.inputs.output_prefix)
        outputs["warped_image"] = f"{out_base}Warped.nii.gz"
        outputs["inverse_warped_image"] = f"{out_base}InverseWarped.nii.gz"
        outputs["out_matrix"] = f"{out_base}0GenericAffine.mat"
        if self.inputs.transform_type not in ("t", "r", "a"):
            outputs["forward_warp_field"] = f"{out_base}1Warp.nii.gz"
            outputs["inverse_warp_field"] = f"{out_base}1InverseWarp.nii.gz"
        return outputs
