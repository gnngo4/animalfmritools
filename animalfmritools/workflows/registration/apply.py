from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import ConvertWarp, ConvertXFM
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from animalfmritools.interfaces.apply_bold_to_anat import ApplyBoldToAnat


def init_merge_bold_to_template_trans(
    name: str = "merge_bold_to_template_transforms_wf",
) -> Workflow:
    """
    Merge transforms
    1) Dbold to Dboldtemplate
    2) Dboldtemplate sdc warp
    3) UDbold to UDboldtemplate
    4) UDboldtemplate to anat
    5) anat to template
    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "regridded_anat",
                "regridded_template",
                "Dbold_to_Dboldtemplate_aff",  # 1
                "Dboldtemplate_sdc_warp",  # 2
                "UDbold_to_UDboldtemplate_aff",  # 3
                "UDboldtemplate_to_anat_aff",  # 4
                "anat_to_template_warp",  # 5
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["Dbold_to_template_warp"]),
        name="outputnode",
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
        (merge_1_5, outputnode, [("out_file", "Dbold_to_template_warp")]),
    ])
    # fmt: on

    return workflow


def init_trans_bold_to_template_wf(
    reg_quick: bool = False,
    name: str = "transform_bold_to_template_wf",
) -> Workflow:
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold",
                "regridded_anat",
                "regridded_template",
                "Dbold_hmc_affs",
                "Dbold_to_Dboldtemplate_aff",
                "Dboldtemplate_sdc_warp",
                "UDbold_to_UDboldtemplate_aff",
                "UDboldtemplate_to_anat_aff",
                "anat_to_template_warp",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_template_space"]),
        name="outputnode",
    )

    merge_bold_to_template_trans = init_merge_bold_to_template_trans()

    apply_bold_to_template = pe.Node(ApplyBoldToAnat(debug=reg_quick), name="apply_transformations")

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
        (merge_bold_to_template_trans, apply_bold_to_template, [("outputnode.Dbold_to_template_warp", "bold_to_anat_warp")]),
        (inputnode, apply_bold_to_template, [
            ("bold", "bold_path"),
            ("Dbold_hmc_affs", "hmc_mats"),
            ("regridded_template", "anat_resampled"),
        ]),
        (apply_bold_to_template, outputnode, [("t1_bold_path", "bold_template_space")])
    ])
    # fmt: on

    return workflow
