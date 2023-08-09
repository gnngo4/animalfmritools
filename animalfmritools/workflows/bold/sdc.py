from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from animalfmritools.interfaces.rescale_nifti import RescaleNifti


def build_list_with_strings(e1, e2):
    return [e1, e2]


def add_string_to_list(in_list, element):
    in_list.append(element)
    return in_list


def init_bold_sdc_wf(
    forward_pe_json: Path,
    reverse_pe_json: Path,
    RESCALE_FACTOR=0.1,
    name: str = "bold_sdc_wf",
):
    from nipype.interfaces.fsl import Merge
    from nipype.interfaces.fsl.maths import BinaryMaths
    from nipype.interfaces.fsl.utils import Split
    from nipype.interfaces.utility import Function

    from animalfmritools.interfaces.fsl_topup import TOPUP, TOPUPAcqParams

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(["forward_pe", "reverse_pe"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(["sdc_warp", "sdc_affine", "sdc_bold"]),
        name="outputnode",
    )

    rescale_fwd_pe = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_fwd_pe")
    rescale_rev_pe = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_rev_pe")

    init_topup_list = pe.Node(
        interface=Function(
            input_name=["e1", "e2"],
            output_names=["out_list"],
            function=build_list_with_strings,
        ),
        name="init_topup_list",
    )

    init_topup_metadata_list = pe.Node(
        interface=Function(
            input_names=["e1", "e2"],
            output_names=["out_list"],
            function=build_list_with_strings,
        ),
        name="init_topup_metadata_list",
    )
    init_topup_metadata_list.inputs.e1 = forward_pe_json
    init_topup_metadata_list.inputs.e2 = reverse_pe_json

    topup_merge = pe.Node(Merge(dimension="t", output_type="NIFTI_GZ"), name="topup_merge")

    topup_get_acqparams = pe.Node(TOPUPAcqParams(), name="topup_get_acqparams_file")

    topup = pe.Node(TOPUP(output_type="NIFTI_GZ"), name="topup")

    topup_split = pe.Node(Split(dimension="t", out_base_name="split_bold_"), name="topup_split_corrected")

    topup_rescale_corrected = pe.Node(RescaleNifti(rescale_factor=1 / RESCALE_FACTOR), name="topup_rescale_corrected")
    topup_rescale_warp_res = pe.Node(
        RescaleNifti(rescale_factor=1 / RESCALE_FACTOR),
        name="topup_rescale_warp_resolution",
    )
    topup_rescale_warp_mag = pe.Node(
        BinaryMaths(operand_value=1 / RESCALE_FACTOR, operation="mul"),
        name="topup_rescale_warp_magnitude",
    )

    # fmt: off
    workflow.connect([
        (inputnode, rescale_fwd_pe, [("forward_pe", "nifti_path")]),
        (inputnode, rescale_rev_pe, [("reverse_pe", "nifti_path")]),
        (rescale_fwd_pe, init_topup_list, [("rescaled_path", "e1")]),
        (rescale_rev_pe, init_topup_list, [("rescaled_path", "e2")]),
        (init_topup_list, topup_merge, [("out_list", "in_files")]),
        (init_topup_metadata_list, topup_get_acqparams, [("out_list", "nifti_list")]),
        (topup_get_acqparams, topup, [("acqparams_txt", "encoding_file")]),
        (topup_merge, topup, [("merged_file", "in_file")]),
        (topup, topup_rescale_corrected, [("out_corrected", "nifti_path")]),
        (topup_rescale_corrected, topup_split, [("rescaled_path", "in_file")]),
        (topup, outputnode, [("out_vol_1_aff", "sdc_affine")]),
        (topup, topup_rescale_warp_res, [("out_vol_1_warp", "nifti_path")]),
        (topup_rescale_warp_res, topup_rescale_warp_mag, [("rescaled_path", "in_file")]),
        (topup_rescale_warp_mag, outputnode, [("out_file", "sdc_warp")]),
        (topup_split, outputnode, [(("out_files", _get_split_volume, 0), "sdc_bold")]),
    ])
    # fmt: on

    return workflow


def _get_split_volume(out_files, vol_idx):
    return out_files[vol_idx]
