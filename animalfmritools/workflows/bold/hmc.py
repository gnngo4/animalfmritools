from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import MCFLIRT
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.confounds import NormalizeMotionParams


def init_bold_hmc_wf(
    name: str = "bold_hmc_wf",
) -> Workflow:
    """Build a workflow to apply head-motion correction (hmc) on a BOLD run.

    Uses FSL's MCFLIRT to perform hmc.

    Args:
        name (str): Name of workflow. (default: "bold_hmc_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        bold: BOLD run
        reference: BOLD reference image

    Workflow Outputs:
        hmc_mats: Transformation matrices of each BOLD volume to the BOLD reference image
        rmsd_file: Relative displacement parameters
        movpar_file: Text file with motion parameters
    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(["bold", "reference"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(["hmc_mats", "rmsd_file", "movpar_file"]),
        name="outputnode",
    )

    hmc = pe.Node(
        MCFLIRT(save_mats=True, save_plots=True, save_rms=True),
        name="mcflirt",
    )
    normalize_motion = pe.Node(
        NormalizeMotionParams(format="FSL"),
        name="normalize_motion_parameters",
    )

    # fmt: off
    workflow.connect([
        (inputnode, hmc, [
            ("bold", "in_file"),
            ("reference", "ref_file"),
        ]),
        (hmc, outputnode, [
            ("mat_file", "hmc_mats"),
            (("rms_files", _pick_rel), "rmsd_file"),
        ]),
        (hmc, normalize_motion,[("par_file", "in_file")]), # NOTE: par_file are rescaled by a factor of 10 due to the `rescale_bold` step
        (normalize_motion, outputnode, [("out_file", "movpar_file")]),
    ])
    # fmt: on

    return workflow


def _pick_rel(rms_files):
    return rms_files[-1]
