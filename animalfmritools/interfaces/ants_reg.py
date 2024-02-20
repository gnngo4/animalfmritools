import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    Str,
    TraitedSpec,
    traits,
)


class RegistrationSyNBaseInputSpec(CommandLineInputSpec):
    dimension = traits.Enum(3, 2, argstr="-d %d", desc="Image dimension (2 or 3)", usedefault=True)
    fixed_image = File(argstr="-f %s", desc="Fixed (or reference image) (.nii.gz)", mandatory=True, exists=True)
    moving_image = File(argstr="-m %s", desc="Moving image (.nii.gz)", mandatory=True, exists=True)
    output_prefix = Str("transform", argstr="-o %s", desc="Prefix of outputs", usedefault=True)
    transform_type = traits.Enum(
        "s", "t", "r", "a", "sr", "b", "br", argstr="-t %s", desc="Transform type", usedefault=True
    )


class RegistrationSyNBaseOutputSpec(TraitedSpec):
    warped_image = File(exists=True, desc="Warped image")
    inverse_warped_image = File(exists=True, desc="Inverse warped image")
    out_matrix = File(exists=True, desc="Affine matrix")
    forward_warp_field = File(exists=True, desc="Forward warp field")
    inverse_warp_field = File(exists=True, desc="Inverse warp field")


class RegistrationSyNBase(CommandLine):
    _cmd = "antsRegistrationSyN.sh"
    input_spec = RegistrationSyNBaseInputSpec
    output_spec = RegistrationSyNBaseOutputSpec

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


class RegistrationSyN(RegistrationSyNBase):
    pass


class RegistrationSyNQuick(RegistrationSyNBase):
    _cmd = "antsRegistrationSyNQuick.sh"
