import os

import numpy as np
from nipype.interfaces.base import (
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.interfaces.fsl.utils import split_filename
from traits.api import List

OUTPATHS = {
    "TOPUP_VOL_1_DFM": "warpfield_01.nii.gz",
    "TOPUP_VOL_1_AFF": "xfm_01.mat",
    "TOPUPAcqParams": "acqparams.txt",
}


### TOPUP - taken from nipype and customized to include (1) warp fields [.nii.gz] and (2) hmc xfm [.mat]
class TOPUPInputSpec(FSLCommandInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="name of 4D file with images",
        argstr="--imain=%s",
    )
    encoding_file = File(
        exists=True,
        mandatory=True,
        xor=["encoding_direction"],
        desc="name of text file with PE directions/times",
        argstr="--datain=%s",
    )
    encoding_direction = traits.List(
        traits.Enum("y", "x", "z", "x-", "y-", "z-"),
        mandatory=True,
        xor=["encoding_file"],
        requires=["readout_times"],
        argstr="--datain=%s",
        desc=("encoding direction for automatic " "generation of encoding_file"),
    )
    readout_times = InputMultiPath(
        traits.Float,
        requires=["encoding_direction"],
        xor=["encoding_file"],
        mandatory=True,
        desc=("readout times (dwell times by # " "phase-encode steps minus 1)"),
    )
    out_base = File(
        desc=("base-name of output files (spline " "coefficients (Hz) and movement parameters)"),
        name_source=["in_file"],
        name_template="%s_base",
        argstr="--out=%s",
        hash_files=False,
    )
    out_field = File(
        argstr="--fout=%s",
        hash_files=False,
        name_source=["in_file"],
        name_template="%s_field",
        desc="name of image file with field (Hz)",
    )
    out_corrected = File(
        argstr="--iout=%s",
        hash_files=False,
        name_source=["in_file"],
        name_template="%s_corrected",
        desc="name of 4D image file with unwarped images",
    )
    out_logfile = File(
        argstr="--logout=%s",
        desc="name of log-file",
        name_source=["in_file"],
        name_template="%s_topup.log",
        keep_extension=True,
        hash_files=False,
    )

    # TODO: the following traits admit values separated by commas, one value
    # per registration level inside topup.
    warp_res = traits.Float(
        10.0,
        argstr="--warpres=%f",
        desc=("(approximate) resolution (in mm) of warp " "basis for the different sub-sampling levels" "."),
    )
    subsamp = traits.Int(1, argstr="--subsamp=%d", desc="sub-sampling scheme")
    fwhm = traits.Float(8.0, argstr="--fwhm=%f", desc="FWHM (in mm) of gaussian smoothing kernel")
    config = traits.String(
        "b02b0.cnf",
        argstr="--config=%s",
        usedefault=True,
        desc=("Name of config file specifying command line " "arguments"),
    )
    dfout = traits.String(
        "warpfield",
        argstr="--dfout=%s",
        usedefault=True,
        desc=("outputname for warpfields"),
    )
    rbmout = traits.String(
        "xfm",
        argstr="--rbmout=%s",
        usedefault=True,
        desc=("outputname for affine transformation"),
    )
    max_iter = traits.Int(5, argstr="--miter=%d", desc="max # of non-linear iterations")
    reg_lambda = traits.Float(
        1.0,
        argstr="--miter=%0.f",
        desc=("lambda weighting value of the " "regularisation term"),
    )
    ssqlambda = traits.Enum(
        1,
        0,
        argstr="--ssqlambda=%d",
        desc=(
            "Weight lambda by the current value of the "
            "ssd. If used (=1), the effective weight of "
            "regularisation term becomes higher for the "
            "initial iterations, therefore initial steps"
            " are a little smoother than they would "
            "without weighting. This reduces the "
            "risk of finding a local minimum."
        ),
    )
    regmod = traits.Enum(
        "bending_energy",
        "membrane_energy",
        argstr="--regmod=%s",
        desc=(
            "Regularisation term implementation. Defaults "
            "to bending_energy. Note that the two functions"
            " have vastly different scales. The membrane "
            "energy is based on the first derivatives and "
            "the bending energy on the second derivatives. "
            "The second derivatives will typically be much "
            "smaller than the first derivatives, so input "
            "lambda will have to be larger for "
            "bending_energy to yield approximately the same"
            " level of regularisation."
        ),
    )
    estmov = traits.Enum(1, 0, argstr="--estmov=%d", desc="estimate movements if set")
    minmet = traits.Enum(
        0,
        1,
        argstr="--minmet=%d",
        desc=("Minimisation method 0=Levenberg-Marquardt, " "1=Scaled Conjugate Gradient"),
    )
    splineorder = traits.Int(
        3,
        argstr="--splineorder=%d",
        desc=("order of spline, 2->Qadratic spline, " "3->Cubic spline"),
    )
    numprec = traits.Enum(
        "double",
        "float",
        argstr="--numprec=%s",
        desc=("Precision for representing Hessian, double " "or float."),
    )
    interp = traits.Enum(
        "spline",
        "linear",
        argstr="--interp=%s",
        desc="Image interpolation model, linear or spline.",
    )
    scale = traits.Enum(
        0,
        1,
        argstr="--scale=%d",
        desc=("If set (=1), the images are individually scaled" " to a common mean"),
    )
    regrid = traits.Enum(
        1,
        0,
        argstr="--regrid=%d",
        desc=("If set (=1), the calculations are done in a " "different grid"),
    )


class TOPUPOutputSpec(TraitedSpec):
    out_fieldcoef = File(exists=True, desc="file containing the field coefficients")
    out_movpar = File(exists=True, desc="movpar.txt output file")
    out_enc_file = File(desc="encoding directions file output for applytopup")
    out_field = File(desc="name of image file with field (Hz)")
    out_corrected = File(desc="name of 4D image file with unwarped images")
    out_logfile = File(desc="name of log-file")
    out_vol_1_warp = File(desc="name of SDC warp-field for first volue")
    out_vol_1_aff = File(desc="name of affine transform for first volume")


class TOPUP(FSLCommand):
    """
    Interface for FSL topup, a tool for estimating and correcting
    susceptibility induced distortions. See FSL documentation for
    `reference <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP>`_,
    `usage examples
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup>`_,
    and `exemplary config files
    <https://github.com/ahheckel/FSL-scripts/blob/master/rsc/fsl/fsl4/topup/b02b0.cnf>`_.

    Examples
    --------

    >>> from nipype.interfaces.fsl import TOPUP
    >>> topup = TOPUP()
    >>> topup.inputs.in_file = "b0_b0rev.nii"
    >>> topup.inputs.encoding_file = "topup_encoding.txt"
    >>> topup.inputs.output_type = "NIFTI_GZ"
    >>> topup.cmdline #doctest: +ELLIPSIS
    'topup --config=b02b0.cnf --datain=topup_encoding.txt \
--imain=b0_b0rev.nii --out=b0_b0rev_base --iout=b0_b0rev_corrected.nii.gz \
--fout=b0_b0rev_field.nii.gz --logout=b0_b0rev_topup.log'
    >>> res = topup.run() # doctest: +SKIP

    """

    _cmd = "topup"
    input_spec = TOPUPInputSpec
    output_spec = TOPUPOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == "encoding_direction":
            return trait_spec.argstr % self._generate_encfile()
        if name == "out_base":
            path, name, ext = split_filename(value)
            if path != "":
                if not os.path.exists(path):
                    raise ValueError("out_base path must exist if provided")
        return super(TOPUP, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = super(TOPUP, self)._list_outputs()
        del outputs["out_base"]
        base_path = None
        if isdefined(self.inputs.out_base):
            base_path, base, _ = split_filename(self.inputs.out_base)
            if base_path == "":
                base_path = None
        else:
            base = split_filename(self.inputs.in_file)[1] + "_base"
        outputs["out_fieldcoef"] = self._gen_fname(base, suffix="_fieldcoef", cwd=base_path)
        outputs["out_movpar"] = self._gen_fname(base, suffix="_movpar", ext=".txt", cwd=base_path)

        if isdefined(self.inputs.encoding_direction):
            outputs["out_enc_file"] = self._get_encfilename()
        outputs["out_vol_1_warp"] = os.path.abspath(OUTPATHS["TOPUP_VOL_1_DFM"])
        outputs["out_vol_1_aff"] = os.path.abspath(OUTPATHS["TOPUP_VOL_1_AFF"])

        return outputs

    def _get_encfilename(self):
        out_file = os.path.join(os.getcwd(), ("%s_encfile.txt" % split_filename(self.inputs.in_file)[1]))
        return out_file

    def _generate_encfile(self):
        """Generate a topup compatible encoding file based on given directions"""
        out_file = self._get_encfilename()
        durations = self.inputs.readout_times
        if len(self.inputs.encoding_direction) != len(durations):
            if len(self.inputs.readout_times) != 1:
                raise ValueError(("Readout time must be a float or match the" "length of encoding directions"))
            durations = durations * len(self.inputs.encoding_direction)

        lines = []
        for idx, encdir in enumerate(self.inputs.encoding_direction):
            direction = 1.0
            if encdir.endswith("-"):
                direction = -1.0
            line = [float(val[0] == encdir[0]) * direction for val in ["x", "y", "z"]] + [durations[idx]]
            lines.append(line)
        np.savetxt(out_file, np.array(lines), fmt="%d %d %d %.8f")
        return out_file

    def _overload_extension(self, value, name=None):
        if name == "out_base":
            return value
        return super(TOPUP, self)._overload_extension(value, name)


### TOPUPAcqParams
def _TOPUPAcqParams(nifti_list, out_path=OUTPATHS["TOPUPAcqParams"]):
    import json

    TOPUP_KEYS = ["EchoTrainLength", "PixelBandwidth"]
    json_list = [str(i).replace(".nii.gz", ".json") for i in nifti_list]
    acqparams_txt = out_path
    with open(acqparams_txt, "w") as f:
        for p in json_list:
            assert "dir-" in p, f"{p} does not contain [dir-]"
            # Load metadata
            try:
                with open(p, "r") as json_f:
                    data = json.load(json_f)
            except Exception:
                data = {}
            # Check if required metadata can be found
            for k in TOPUP_KEYS:
                if k not in data.keys():
                    print(f"[{k}] metadata is missing from .json")
                    total_readout_time = 0.01
                else:
                    echo_spacing = 1 / (data["PixelBandwidth"])
                    n_echos = data["EchoTrainLength"]
                    total_readout_time = n_echos * echo_spacing
            pedir = p.split("dir-")[-1].split("_")[0]
            if pedir == "PA":
                pedir_encoding = ["0", "1", "0", f"{total_readout_time:.5f}"]
            elif pedir == "AP":
                pedir_encoding = ["0", "-1", "0", f"{total_readout_time:.5f}"]
            elif pedir == "RL":
                pedir_encoding = ["1", "0", "0", f"{total_readout_time:.5f}"]
            elif pedir == "LR":
                pedir_encoding = ["-1", "0", "0", f"{total_readout_time:.5f}"]
            elif pedir == "SI":
                pedir_encoding = ["0", "0", "1", f"{total_readout_time:.5f}"]
            elif pedir == "IS":
                pedir_encoding = ["0", "0", "-1", f"{total_readout_time:.5f}"]
            else:
                raise NotImplementedError("This functionality is not implemented yet.")
            f.write(" ".join(pedir_encoding) + "\n")

    return acqparams_txt


class TOPUPAcqParamsInputSpec(TraitedSpec):
    nifti_list = List(desc="list of nifti paths", mandatory=True)


class TOPUPAcqParamsOutputSpec(TraitedSpec):
    acqparams_txt = File(exists=True, desc="TOPUP acquisition parameter text file")


class TOPUPAcqParams(SimpleInterface):
    input_spec = TOPUPAcqParamsInputSpec
    output_spec = TOPUPAcqParamsOutputSpec

    def _run_interface(self, runtime):
        _TOPUPAcqParams(
            self.inputs.nifti_list,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["acqparams_txt"] = os.path.abspath(OUTPATHS["TOPUPAcqParams"])

        return outputs
