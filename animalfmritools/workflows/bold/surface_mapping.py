from typing import Any, Dict

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from animalfmritools.interfaces.workbench import (
    CreateDenseTimeseries,
    VolumeToSurfaceMapping,
)

BOLD_DTSERIES = "DenseTimeseries.dtseries.nii"


def init_bold_surf_wf(
    metadata: Dict[str, Any],
    BOLD_DTSERIES: str = BOLD_DTSERIES,
    name: str = "bold_surf_wf",
) -> Workflow:
    """Build a workflow to project template-normalized BOLD data onto a surface.

    Args:
        metadata (dict): Metadata of the BOLD run (must include the repetition time [or TR]).
        BOLD_DTSERIES (str): Name of relative path to the temporary file associated with the surface-projected BOLD run. (default: `BOLD_DTSERIES`)
        name (str): Name of workflow. (default: "bold_surf_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        bold_nifti: Template-normalized BOLD run (.nii.gz)
        lh_midthickness: Left midthickness surface (.surf.gii)
        rh_midthickness: Right midthickness surface (.surf.gii)
        lh_white: Left white matter surface (.surf.gii)
        rh_white: Right white matter surface (.surf.gii)
        lh_pial: Left pial surface (.surf.gii)
        rh_pial: Right pial surface (.surf.gii)
        lh_cortex: Left cortex mask (.func.gii)
        rh_cortex: Right cortex mask (.func.gii)

    Workflow Outputs:
        bold_dtseries: Template and surface projected BOLD run (.dtseries.nii)

    See also:
        - :func: `~animalfmritools.interfaces.workbench.CreateDenseTimeseries`
        - :func: `~animalfmritools.interfaces.workbench.VolumeToSurfaceMapping`
    """

    workflow = Workflow(name=name)

    BOLD_SURF_INPUTS = [
        "bold_nifti",
        "lh_midthickness",
        "rh_midthickness",
        "lh_white",
        "rh_white",
        "lh_pial",
        "rh_pial",
        "lh_cortex",
        "rh_cortex",
    ]

    inputnode = pe.Node(
        niu.IdentityInterface(BOLD_SURF_INPUTS),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(["bold_dtseries"]),
        name="outputnode",
    )

    surface_mapping_nodes = {}
    for hemi in ['lh', 'rh']:
        surface_mapping_nodes[hemi] = pe.Node(
            VolumeToSurfaceMapping(bold_gifti=f"{hemi}_bold.func.gii"), name=f"{hemi}_surface_mapping"
        )
        # fmt: off
        workflow.connect([
            (inputnode, surface_mapping_nodes[hemi], [
                ("bold_nifti", "bold_nifti"),
                (f"{hemi}_midthickness", "surf_midthickness"),
                (f"{hemi}_white", "surf_white"),
                (f"{hemi}_pial", "surf_pial"),
            ]),
        ])
        # fmt: on

    create_dense_timeseries = pe.Node(
        CreateDenseTimeseries(bold_dtseries=BOLD_DTSERIES, TR=metadata["RepetitionTime"]),
        name="create_dense_timeseries",
    )

    # fmt: off
    workflow.connect([
        (inputnode, create_dense_timeseries, [
            ("lh_cortex", "left_roi"),
            ("rh_cortex", "right_roi"),
        ]),
        (surface_mapping_nodes['lh'], create_dense_timeseries, [("bold_gifti", "left_metric")]),
        (surface_mapping_nodes['rh'], create_dense_timeseries, [("bold_gifti", "right_metric")]),
        (create_dense_timeseries, outputnode, [("bold_dtseries", "bold_dtseries")]),
    ])
    # fmt: on

    return workflow
