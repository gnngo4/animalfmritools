import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)

CMDBASE = "wb_command"


class VolumeToSurfaceMappingInputSpec(CommandLineInputSpec):
    bold_nifti = File(
        exists=True,
        mandatory=True,
        argstr="-volume-to-surface-mapping %s",
        position=0,
        desc="input bold (.nii.gz)",
    )
    surf_midthickness = File(argstr="%s", position=1, desc="midthickness surface", mandatory=True, exists=True)
    bold_gifti = File(argstr="%s", position=2, desc="output bold (.func.gii)", mandatory=True)
    surf_white = File(argstr="-ribbon-constrained %s", position=3, desc="white surface", mandatory=True, exists=True)
    surf_pial = File(argstr="%s", position=4, desc="pial surface", mandatory=True, exists=True)


class VolumeToSurfaceMappingOutputSpec(TraitedSpec):
    bold_gifti = File(desc="output bold (.func.gii)")


class VolumeToSurfaceMapping(CommandLine):
    _cmd = CMDBASE
    input_spec = VolumeToSurfaceMappingInputSpec
    output_spec = VolumeToSurfaceMappingOutputSpec

    def _list_outputs(self):
        _outputs = {
            "bold_gifti": os.path.abspath(self.inputs.bold_gifti),
        }

        return _outputs


class CreateDenseTimeseriesInputSpec(CommandLineInputSpec):
    bold_dtseries = File(
        mandatory=True,
        argstr="-cifti-create-dense-timeseries %s",
        position=0,
        desc="output bold (.dtseries.nii)",
    )
    left_metric = File(
        argstr="-left-metric %s",
        position=1,
        desc="input bold [left hemisphere] (.func.gii)",
        mandatory=True,
        exists=True,
    )
    left_roi = File(
        argstr="-roi-left %s",
        position=2,
        desc="input roi of surface [left hemisphere] (.func.gii)",
        mandatory=True,
        exists=True,
    )
    right_metric = File(
        argstr="-right-metric %s",
        position=3,
        desc="input bold [right hemisphere] (.func.gii)",
        mandatory=True,
        exists=True,
    )
    right_roi = File(
        argstr="-roi-right %s",
        position=4,
        desc="input roi of surface [right hemisphere] (.func.gii)",
        mandatory=True,
        exists=True,
    )
    TR = traits.Float(argstr="-timestep %f", position=5, mandatory=True, desc="set TR of the output bold")


class CreateDenseTimeseriesOutputSpec(TraitedSpec):
    bold_dtseries = File(desc="output bold (.dtseries.nii)")


class CreateDenseTimeseries(CommandLine):
    _cmd = CMDBASE
    input_spec = CreateDenseTimeseriesInputSpec
    output_spec = CreateDenseTimeseriesOutputSpec

    def _list_outputs(self):
        _outputs = {
            "bold_dtseries": os.path.abspath(self.inputs.bold_dtseries),
        }

        return _outputs
