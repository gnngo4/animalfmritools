from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)

import os


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
    _cmd = "c3d_affine_tool2"
    input_spec = C3dAffineToolInputSpec
    output_spec = C3dAffineToolOutputSpec

    def _list_outputs(self):
        _outputs = {
            "fsl_transform": os.path.abspath(self.inputs.fsl_transform),
        }

        return _outputs
