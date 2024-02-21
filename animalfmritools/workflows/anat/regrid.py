from typing import Optional

from nipype.interfaces import utility as niu
from nipype.interfaces.ants import ApplyTransforms, N4BiasFieldCorrection
from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.fsl.maths import UnaryMaths
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nibabel import GenerateSamplingReference


def init_regrid_anat_to_bold_wf(
    regrid_to_bold: bool = True,
    name: str = "regrid_anat_to_bold_wf",
) -> Workflow:
    """Build a workflow to regrid an anatomical to a BOLD image.

    This workflow includes N4 Bias Field Correction applied to the anatomical image.

    Args:
        regrid_to_bold (bool): True will match the resolution of the anatomical to the BOLD image. False does nothing. (default: True)
        name (str): Name of workflow. (default: "regrid_anat_to_bold_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        anat: Anatomical
        bold: BOLD reference

    Workflow Outputs:
        regridded_anat: Regridded anatomical
    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["anat", "bold"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["regridded_anat"]),
        name="outputnode",
    )

    n4_anat = pe.Node(N4BiasFieldCorrection(), name="n4_anat")

    if regrid_to_bold:
        regrid_anat = pe.Node(GenerateSamplingReference(), name="regrid_anat")
        # fmt: off
        workflow.connect([
            (inputnode, regrid_anat, [("bold", "moving_image")]),
            (n4_anat, regrid_anat, [("output_image", "fixed_image")]),
            (regrid_anat, outputnode, [("out_file", "regridded_anat")]),
        ])
        # fmt: on
    else:
        # fmt: off
        workflow.connect([
            (n4_anat, outputnode, [("output_image", "regridded_anat")])
        ])
        # fmt: on

    # fmt: off
    workflow.connect([
        (inputnode, n4_anat, [("anat", "input_image")]),
    ])
    # fmt: on

    return workflow


def init_regrid_template_to_bold_wf(
    force_isotropic: Optional[float] = None,
    name: str = "regrid_template_to_bold_wf",
) -> Workflow:
    """Build a workflow to regrid an template to a BOLD image.

    Args:
        force_isotropic (Optional[float]): If a float value is specified, it will enforce the template to the desired isotropic resolution. Using None will align the template resolution with that of the BOLD image. (default: None)
        name (str): Name of workflow. (default: "regrid_template_to_bold_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        template: Template
        template_gm: Template gray matter mask
        template_wm: Template white matter mask
        template_csf: Template CSF mask
        bold: BOLD reference

    Workflow Outputs:
        regridded_template: Regridded template
        regridded_template_mask: Regridded template mask
        regridded_template_tpms: Regridded merged masks of the template
    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "template",
                "template_gm",
                "template_wm",
                "template_csf",
                "bold",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "regridded_template",
                "regridded_template_mask",
                "regridded_template_tpms",
            ]
        ),
        name="outputnode",
    )

    template_buffer = pe.Node(niu.IdentityInterface(["template"]), name="template_buffer")
    if force_isotropic is None:
        regrid_template = pe.Node(GenerateSamplingReference(), name="regrid_template")
        # fmt: off
        workflow.connect([
            (inputnode, regrid_template, [
                ("bold", "moving_image"),
                ("template", "fixed_image")
            ]),
            (regrid_template, outputnode, [("out_file", "regridded_template")]),
            (regrid_template, template_buffer, [("out_file", "template")]),
        ])
        # fmt: on
    else:
        flirt_force_iso = pe.Node(FLIRT(apply_isoxfm=force_isotropic), name="force_isotropic")
        # fmt: off
        workflow.connect([
            (inputnode, flirt_force_iso, [
                ("template", "in_file"),
                ("template", "reference"),
            ]),
            (flirt_force_iso, outputnode, [("out_file", "regridded_template")]),
            (flirt_force_iso, template_buffer, [("out_file", "template")]),
        ])
        # fmt: on

    binarize_template = pe.Node(UnaryMaths(operation="bin"), name="binarize_template")
    regrid_template_gm = pe.Node(ApplyTransforms(transforms="identity"), name="regrid_template_gm")
    regrid_template_wm = pe.Node(ApplyTransforms(transforms="identity"), name="regrid_template_wm")
    regrid_template_csf = pe.Node(ApplyTransforms(transforms="identity"), name="regrid_template_csf")
    merge_template_tpms = pe.Node(
        Function(
            input_names=["input_1", "input_2", "input_3"],
            output_names=["output_list"],
            function=_listify_three_inputs,
        ),
        name="merge_template_tpms",
    )

    # fmt: off
    workflow.connect([
        (inputnode, regrid_template_gm, [("template_gm", "input_image")]),
        (inputnode, regrid_template_wm, [("template_wm", "input_image")]),
        (inputnode, regrid_template_csf, [("template_csf", "input_image")]),
        (template_buffer, regrid_template_gm, [("template", "reference_image")]),
        (template_buffer, regrid_template_wm, [("template", "reference_image")]),
        (template_buffer, regrid_template_csf, [("template", "reference_image")]),
        (regrid_template_gm, merge_template_tpms, [("output_image", "input_1")]),
        (regrid_template_wm, merge_template_tpms, [("output_image", "input_2")]),
        (regrid_template_csf, merge_template_tpms, [("output_image", "input_3")]),
        (merge_template_tpms, outputnode, [("output_list", "regridded_template_tpms")]),
        (template_buffer, binarize_template, [("template", "in_file")]),
        (binarize_template, outputnode, [("out_file", "regridded_template_mask")]),
    ])
    # fmt: on

    return workflow


def _listify_three_inputs(input_1, input_2, input_3):
    return [input_1, input_2, input_3]
