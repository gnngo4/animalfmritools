from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from animalfmritools.interfaces.itk_to_fsl import C3dAffineTool, ConvertITKtoFSLWarp


def init_itk_to_fsl_affine_wf(name="itk_to_fsl_affine_wf"):
    """Build a workflow to convert affine transformation from itk-to-fsl format.

    Args:
        name (str): Name of workflow. (default: "itk_to_fsl_affine_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        itk_affine: Affine transformation (itk format)
        source: Moving image
        reference: Reference image

    Workflow Outputs:
        fsl_affine: Affine transformation (fsl format)
    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(["itk_affine", "source", "reference"]),
        name="inputnode",
    )

    outputnode = pe.Node(niu.IdentityInterface(["fsl_affine"]), name="outputnode")

    itk_to_fsl = pe.Node(C3dAffineTool(ras2fsl=True), name="itk_to_fsl_affine")
    itk_to_fsl.inputs.fsl_transform = "fsl_affine.mat"

    # Connect
    # fmt: off
    workflow.connect([
        (inputnode, itk_to_fsl, [
            ("source", "source_file"),
            ("reference", "reference_file"),
            ("itk_affine", "itk_transform"),
        ]),
        (itk_to_fsl, outputnode, [("fsl_transform", "fsl_affine")]),
    ])
    # fmt: on

    return workflow


def init_itk_to_fsl_warp_wf(name="itk_to_fsl_warp_wf"):
    """Build a workflow to convert warp transformation from itk-to-fsl format.

    Args:
        name (str): Name of workflow. (default: "itk_to_fsl_warp_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        itk_warp: Warp transformation (itk format)
        reference: Reference image

    Workflow Outputs:
        fsl_warp: Warp transformation (fsl format)
    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(["itk_warp", "fsl_warp", "reference"]),
        name="inputnode",
    )

    outputnode = pe.Node(niu.IdentityInterface(["fsl_warp"]), name="outputnode")

    itk_to_fsl = pe.Node(ConvertITKtoFSLWarp(), name="itk_to_fsl_warp")
    itk_to_fsl.inputs.fsl_warp = "fsl_warp.nii.gz"

    # Connect
    # fmt: off
    workflow.connect([
        (inputnode, itk_to_fsl, [
            ("itk_warp", "itk_warp"),
            ("reference", "reference"),
        ]),
        (itk_to_fsl, outputnode, [("fsl_warp", "fsl_warp")]),
    ])
    # fmt: on

    return workflow
