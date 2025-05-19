from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import ConvertWarp, ConvertXFM
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from animalfmritools.interfaces.apply_bold_to_anat import ApplyBoldToAnat

SCENARIO_A = [
    "Dbold_to_Dboldtemplate_aff",  # 1
    "Dboldtemplate_sdc_warp",  # 2
    "UDbold_to_UDboldtemplate_aff",  # 3
    "UDboldtemplate_to_anat_aff",  # 4
    "anat_to_template_warp",  # 5
]
SCENARIO_B = [
    "Dbold_to_Dboldtemplate_aff",  # 1
    "Dboldtemplate_to_anat_aff",  # 2
    "anat_to_template_warp",  # 3
]


def init_merge_bold_to_template_trans(
    no_sdc: bool = False,
    use_anat_to_guide: bool = False,
    name: str = "merge_bold_to_template_transforms_wf",
) -> Workflow:
    """Build a workflow to merge all of the transformations together. This excludes head-motion correction.

    Notes:
        [Scenario A] If opposite phase-encoding (PE) BOLD runs are detected, then susceptibility-induced distortion correction (SDC) warps are estimated using FSL's TOPUP.
        Merge transforms
        1) Dbold to Dboldtemplate
        2) Dboldtemplate sdc warp
        3) UDbold to UDboldtemplate
        4) UDboldtemplate to anat [or anat_native]
        5-) anat_native to anat
        5) anat to template
        [Scenario B] If opposite PE BOLD runs are not detected, then no SDC warp is estimated.
        Merge transforms
        1) Dbold to Dboldtemplate
        2) Dboldtemplate to anat [or anat_native]
        3-) anat_native to anat
        3) anat to template

    Args:
        no_sdc (bool): True will enable scenario B. (default: False)
        use_anat_to_guide (bool): True adds extra nodes to connect a anatnative-to-anat transformation. (default: False)
        name (str): Name of workflow. (default: "merge_bold_to_template_transforms_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        regridded_anat: Regridded anatomical
        regridded_template: Regridded template
        anat_native_to_anat_secondary_warp: Anatomical native to anatomical template warp, inputted when `use_anat_to_guide==True`
        Dbold_to_Dboldtemplate_aff: Distorted bold to distorted bold template affine (Scenario A/B)
        Dboldtemplate_sdc_warp: Distorted bold susceptibility distortion correction warp (Scenario A)
        UDbold_to_UDboldtemplate_aff: Undistorted bold to undistorted bold template affine (Scenario A)
        UDboldtemplate_to_anat_aff: Undistorted bold to anatomical (template or native when `use_anat_to_guide==True`) (Scenario A)
        Dboldtemplate_to_anat_aff: Distorted bold to anatomical (template or native when `use_anat_to_guide==True`) (Scenario B)
        anat_to_template_warp: Anatomical (template or native) to template warp (Scenario A/B)

    Workflow Outputs:
        Dbold_to_template_warp: One-shot distorted bold to template warp
    """

    workflow = Workflow(name=name)

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["Dbold_to_template_warp"]),
        name="outputnode",
    )

    INPUTNODE_FILES = ["regridded_anat", "regridded_template"]

    if use_anat_to_guide:
        INPUTNODE_FILES += ["anat_native_to_anat_secondary_warp"]

    # Scenario B
    if no_sdc:
        INPUTNODE_FILES += SCENARIO_B
        inputnode = pe.Node(
            niu.IdentityInterface(fields=INPUTNODE_FILES),
            name="inputnode",
        )
        merge_1_2 = pe.Node(
            ConvertXFM(concat_xfm=True),
            name="merge_steps_1_2",
        )
        merge_1_3 = pe.Node(
            ConvertWarp(output_type="NIFTI_GZ", relwarp=True),
            name="merge_steps_1_3",
        )
        # fmt: off
        workflow.connect([
            (inputnode, merge_1_2, [
                ("Dbold_to_Dboldtemplate_aff", "in_file"),
                ("Dboldtemplate_to_anat_aff", "in_file2")
            ]),
            (merge_1_2, merge_1_3, [
                ("out_file", "premat"),
            ]),
            (inputnode, merge_1_3, [
                ("anat_to_template_warp", "warp1"),
                ("regridded_template", "reference"),
            ]),
            (merge_1_3, outputnode, [("out_file", "Dbold_to_template_warp")])
        ])
        # fmt: on

        if use_anat_to_guide:
            merge_anat_warps = pe.Node(ConvertWarp(output_type="NIFTI_GZ", relwarp=True), name="merge_anat_warps")
            # fmt: off
            workflow.disconnect([(inputnode, merge_1_3, [("anat_to_template_warp", "warp1")])])
            workflow.connect([
                (inputnode, merge_anat_warps, [
                    ("anat_native_to_anat_secondary_warp", "warp1"),
                    ("anat_to_template_warp", "warp2"),
                    ("regridded_template", "reference"),
                ]),
                (merge_anat_warps, merge_1_3, [("out_file", "warp1")]),
            ])
            # fmt: on

    # Scenario A
    else:
        INPUTNODE_FILES += SCENARIO_A
        inputnode = pe.Node(
            niu.IdentityInterface(fields=INPUTNODE_FILES),
            name="inputnode",
        )
        merge_3_4 = pe.Node(
            ConvertXFM(concat_xfm=True),
            name="merge_steps_3_4",
        )
        merge_1_4 = pe.Node(
            ConvertWarp(output_type="NIFTI_GZ", relwarp=True),
            name="merge_steps_1_4",
        )
        merge_1_5 = pe.Node(
            ConvertWarp(output_type="NIFTI_GZ", relwarp=True),
            name="merge_steps_1_5",
        )
        # fmt: off
        workflow.connect([
            (inputnode, merge_3_4, [
                ("UDbold_to_UDboldtemplate_aff", "in_file"),
                ("UDboldtemplate_to_anat_aff", "in_file2")
            ]),
            (inputnode, merge_1_4, [
                ("Dbold_to_Dboldtemplate_aff", "premat"),
                ("Dboldtemplate_sdc_warp", "warp1")
            ]),
            (merge_3_4, merge_1_4, [("out_file", "postmat")]),
            (inputnode, merge_1_4, [("regridded_anat", "reference")]),
            (merge_1_4, merge_1_5, [("out_file", "warp1")]),
            (inputnode, merge_1_5, [
                ("anat_to_template_warp", "warp2"),
                ("regridded_template", "reference")
            ]),
            (merge_1_5, outputnode, [("out_file", "Dbold_to_template_warp")])
        ])
        # fmt: on

        if use_anat_to_guide:
            merge_anat_warps = pe.Node(ConvertWarp(output_type="NIFTI_GZ", relwarp=True), name="merge_anat_warps")
            # fmt: off
            workflow.disconnect([(inputnode, merge_1_5, [("anat_to_template_warp", "warp2")])])
            workflow.connect([
                (inputnode, merge_anat_warps, [
                    ("anat_native_to_anat_secondary_warp", "warp1"),
                    ("anat_to_template_warp", "warp2"),
                    ("regridded_template", "reference"),
                ]),
                (merge_anat_warps, merge_1_5, [("out_file", "warp2")]),
            ])
            # fmt: on

    return workflow


def init_trans_bold_to_template_wf(
    no_sdc: bool = False,
    reg_quick: bool = False,
    num_procs: int = 4,
    use_anat_to_guide: bool = False,
    name: str = "transform_bold_to_template_wf",
) -> Workflow:
    """Build a workflow that transforms BOLD data into standard (or template) space.

    Args:
        no_sdc (bool): True will enable scenario B. (default: False)
        reg_quick (bool): True will output transformation of only the first 10 BOLD volumes to standard space. (default=False)
        use_anat_to_guide (bool): True adds extra nodes to connect a anatnative-to-anat transformation. (default: False)
        name (str): Name of workflow. (default: "transform_bold_to_template_wf")

    Returns:
        Workflow: The constructed workflow.

    Workflow Inputs:
        bold: Unprocessed BOLD data
        regridded_anat: Regridded anatomical
        regridded_template: Regridded template
        Dbold_hmc_affs: Head-motion correction affines generated using MCFLIRT

    Workflow Outputs:
        bold_template_space: Minimally processed and standardized-to-template BOLD data
    """
    workflow = Workflow(name=name)

    INPUTNODE_FILES = ["bold", "regridded_anat", "regridded_template", "Dbold_hmc_affs"]

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_template_space"]),
        name="outputnode",
    )

    merge_bold_to_template_trans = init_merge_bold_to_template_trans(no_sdc=no_sdc, use_anat_to_guide=use_anat_to_guide)
    apply_bold_to_template = pe.Node(ApplyBoldToAnat(debug=reg_quick, num_procs=num_procs), name="apply_transformations")

    if use_anat_to_guide:
        INPUTNODE_FILES += ["anat_native_to_anat_secondary_warp"]

    if no_sdc:
        INPUTNODE_FILES += SCENARIO_B
        inputnode = pe.Node(
            niu.IdentityInterface(fields=INPUTNODE_FILES),
            name="inputnode",
        )
        # fmt: off
        workflow.connect([
            (inputnode, merge_bold_to_template_trans, [
                ("regridded_anat","inputnode.regridded_anat"),
                ("regridded_template","inputnode.regridded_template"),
                ("Dbold_to_Dboldtemplate_aff","inputnode.Dbold_to_Dboldtemplate_aff"),
                ("Dboldtemplate_to_anat_aff","inputnode.Dboldtemplate_to_anat_aff"),
                ("anat_to_template_warp","inputnode.anat_to_template_warp"),
            ]),
        ])
        # fmt: on
    else:
        INPUTNODE_FILES += SCENARIO_A
        inputnode = pe.Node(
            niu.IdentityInterface(fields=INPUTNODE_FILES),
            name="inputnode",
        )
        # fmt: off
        workflow.connect([
            (inputnode, merge_bold_to_template_trans, [
                ("regridded_anat","inputnode.regridded_anat"),
                ("regridded_template","inputnode.regridded_template"),
                ("Dbold_to_Dboldtemplate_aff","inputnode.Dbold_to_Dboldtemplate_aff"),
                ("Dboldtemplate_sdc_warp","inputnode.Dboldtemplate_sdc_warp"),
                ("UDbold_to_UDboldtemplate_aff","inputnode.UDbold_to_UDboldtemplate_aff"),
                ("UDboldtemplate_to_anat_aff","inputnode.UDboldtemplate_to_anat_aff"),
                ("anat_to_template_warp","inputnode.anat_to_template_warp"),
            ]),
        ])
        # fmt: on

    # fmt: off
    workflow.connect([
        (merge_bold_to_template_trans, apply_bold_to_template, [("outputnode.Dbold_to_template_warp", "bold_to_anat_warp")]),
        (inputnode, apply_bold_to_template, [
            ("bold", "bold_path"),
            ("Dbold_hmc_affs", "hmc_mats"),
            ("regridded_template", "anat_resampled"),
        ]),
        (apply_bold_to_template, outputnode, [("t1_bold_path", "bold_template_space")])
    ])
    # fmt: on

    if use_anat_to_guide:
        # fmt: off
        workflow.connect([
            (inputnode, merge_bold_to_template_trans, [
                ("anat_native_to_anat_secondary_warp", "inputnode.anat_native_to_anat_secondary_warp")
            ])
        ])
        # fmt: on

    return workflow
