import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class C3dAffineToolInputSpec(CommandLineInputSpec):
    reference_file = File(
        exists=True,
        mandatory=True,
        argstr="-ref %s",
        position=0,
        desc="reference image",
    )
    source_file = File(argstr="-src %s", position=1, desc="source image")
    itk_transform = File(argstr="-itk %s", position=2, desc="itk transform")
    fsl_transform = File(argstr="-o %s", position=4, desc="fsl transform")
    ras2fsl = traits.Bool(
        mandatory=False,
        argstr="-ras2fsl",
        position=3,
        desc="ras2fsl flag",
    )


class C3dAffineToolOutputSpec(TraitedSpec):
    fsl_transform = File(desc="fsl transform")


class C3dAffineTool(CommandLine):
    _cmd = "c3d_affine_tool"
    input_spec = C3dAffineToolInputSpec
    output_spec = C3dAffineToolOutputSpec

    def _list_outputs(self):
        _outputs = {
            "fsl_transform": os.path.abspath(self.inputs.fsl_transform),
        }

        return _outputs


class ConvertITKtoFSLWarpInputSpec(CommandLineInputSpec):
    itk_warp = File(argstr="-from-itk %s", position=0, desc="ITK warp")
    fsl_warp = File(argstr="-to-fnirt %s", position=1, desc="FSL warp")
    reference = File(argstr="%s", position=2, desc="Reference")


class ConvertITKtoFSLWarpOutputSpec(TraitedSpec):
    fsl_warp = File(desc="FSL warp")


class ConvertITKtoFSLWarp(CommandLine):
    _cmd = "wb_command -convert-warpfield"
    input_spec = ConvertITKtoFSLWarpInputSpec
    output_spec = ConvertITKtoFSLWarpOutputSpec

    def _list_outputs(self):
        _outputs = {
            "fsl_warp": os.path.abspath(self.inputs.fsl_warp),
        }

        return _outputs
