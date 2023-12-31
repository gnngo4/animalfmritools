from nipype.interfaces import utility as niu
from nipype.interfaces.ants import ApplyTransforms, N4BiasFieldCorrection
from nipype.interfaces.fsl.maths import UnaryMaths
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nibabel import GenerateSamplingReference


def init_regrid_anat_to_bold_wf(
    name: str = "regrid_anat_to_bold_wf",
) -> Workflow:
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
    regrid_anat = pe.Node(GenerateSamplingReference(), name="regrid_anat")

    # fmt: off
    workflow.connect([
        (inputnode, n4_anat, [("anat", "input_image")]),
        (inputnode, regrid_anat, [("bold", "moving_image")]),
        (n4_anat, regrid_anat, [("output_image", "fixed_image")]),
        (regrid_anat, outputnode, [("out_file", "regridded_anat")]),
    ])
    # fmt: on

    return workflow


def init_regrid_template_to_bold_wf(
    name: str = "regrid_template_to_bold_wf",
) -> Workflow:
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

    regrid_template = pe.Node(GenerateSamplingReference(), name="regrid_template")
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
        (inputnode, regrid_template, [
            ("bold", "moving_image"),
            ("template", "fixed_image")
        ]),
        (regrid_template, outputnode, [("out_file", "regridded_template")]),
        (inputnode, regrid_template_gm, [("template_gm", "input_image")]),
        (inputnode, regrid_template_wm, [("template_wm", "input_image")]),
        (inputnode, regrid_template_csf, [("template_csf", "input_image")]),
        (regrid_template, regrid_template_gm, [("out_file", "reference_image")]),
        (regrid_template, regrid_template_wm, [("out_file", "reference_image")]),
        (regrid_template, regrid_template_csf, [("out_file", "reference_image")]),
        (regrid_template_gm, merge_template_tpms, [("output_image", "input_1")]),
        (regrid_template_wm, merge_template_tpms, [("output_image", "input_2")]),
        (regrid_template_csf, merge_template_tpms, [("output_image", "input_3")]),
        (merge_template_tpms, outputnode, [("output_list", "regridded_template_tpms")]),
        (regrid_template, binarize_template, [("out_file", "in_file")]),
        (binarize_template, outputnode, [("out_file", "regridded_template_mask")]),
    ])
    # fmt: on

    return workflow


def _listify_three_inputs(input_1, input_2, input_3):
    return [input_1, input_2, input_3]
