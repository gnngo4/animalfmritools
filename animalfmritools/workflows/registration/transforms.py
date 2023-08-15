from typing import List

from fmriprep.workflows.bold.registration import init_fsl_bbr_wf
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection, RegistrationSynQuick
from nipype.interfaces.fsl import FLIRT, MCFLIRT, ApplyWarp, ConvertWarp, ConvertXFM
from nipype.interfaces.fsl.maths import ApplyMask, MeanImage, Threshold
from nipype.interfaces.fsl.utils import Split
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.reportlets.registration import FLIRTRPT, ANTSApplyTransformsRPT

from animalfmritools.workflows.registration.utils import init_itk_to_fsl_affine_wf, init_itk_to_fsl_warp_wf


def connect_n4_nodes(
    inputnode: pe.Node,
    buffer: pe.Node,
    workflow: Workflow,
    fields: List[str],
    n4_reg_flag: bool = False,
) -> Workflow:
    if n4_reg_flag:
        for field in fields:
            n4biasfieldcorrection = pe.Node(N4BiasFieldCorrection(), name=f"n4biasfieldcorrection_{field}")
            # fmt: off
            workflow.connect([
                (inputnode, n4biasfieldcorrection, [(field, "input_image")]),
                (n4biasfieldcorrection, buffer, [("output_image", field)]),
            ])
            # fmt: on
    else:
        for field in fields:
            # fmt: off
            workflow.connect([
                (inputnode, buffer, [(field, field)]),
            ])
            # fmt: on

    return workflow


def init_reg_Dbold_to_Dboldtemplate_wf(
    n4_reg_flag: bool = False,
    name: str = "reg_Dbold_to_Dboldtemplate_wf",
) -> Workflow:
    """
    Dbold: distorted bold (or uncorrected with FSL's TOPUP)
    Dboldtemplate: distorted bold (heuristically select the 1st run as the template)
    NOTE: In this pipeline, FSL's TOPUP is applied to Dboldtemplate
    """

    workflow = Workflow(name=name)

    INPUTNODE_FIELDS = ["Dbold", "Dboldtemplate"]

    inputnode = pe.Node(
        niu.IdentityInterface(fields=INPUTNODE_FIELDS),
        name="inputnode",
    )

    n4_buffer = pe.Node(niu.IdentityInterface(fields=INPUTNODE_FIELDS), name="n4_buffer")

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_report", "fsl_affine"]),
        name="outputnode",
    )

    # Connect `inputnode` to `n4_buffer`
    workflow = connect_n4_nodes(inputnode, n4_buffer, workflow, INPUTNODE_FIELDS, n4_reg_flag=n4_reg_flag)

    reg_Dbold_to_Dboldtemplate = init_fsl_bbr_wf(
        bold2t1w_dof=6,
        use_bbr=False,
        bold2t1w_init="register",
        omp_nthreads=4,
        name="reg_Dbold_to_Dboldtemplate",
    )

    xfm_convert_itk_to_fsl = init_itk_to_fsl_affine_wf(name="itk_to_fsl_affine")

    # fmt: off
    workflow.connect([
        (n4_buffer, reg_Dbold_to_Dboldtemplate, [
            ("Dbold", "inputnode.in_file"),
            ("Dboldtemplate", "inputnode.t1w_brain"),
        ]),
        (reg_Dbold_to_Dboldtemplate, outputnode, [("outputnode.out_report", "out_report")]),
        (reg_Dbold_to_Dboldtemplate, xfm_convert_itk_to_fsl, [("outputnode.itk_bold_to_t1", "inputnode.itk_affine")]),
        (n4_buffer, xfm_convert_itk_to_fsl, [
            ("Dbold", "inputnode.source"),
            ("Dboldtemplate", "inputnode.reference"),
        ]),
        (xfm_convert_itk_to_fsl, outputnode, [("outputnode.fsl_affine", "fsl_affine")]),
    ])
    # fmt: on

    return workflow


def init_reg_UDbold_to_UDboldtemplate_wf(
    n4_reg_flag: bool = False, name: str = "reg_UDbold_to_UDboldtemplate_wf"
) -> Workflow:
    """
    UDbold: undistorted bold (or corrected with FSL's TOPUP)
    UDboldtemplate: undistorted bold (heuristically select the 1st detected PE dir run as the template)
    NOTE: In this pipeline, UDboldtemplate is used for spatial normalization to the animal's anatomical scan (anat)
    """

    workflow = Workflow(name=name)

    INPUTNODE_FIELDS = ["UDbold", "UDboldtemplate"]

    inputnode = pe.Node(
        niu.IdentityInterface(fields=INPUTNODE_FIELDS),
        name="inputnode",
    )

    n4_buffer = pe.Node(niu.IdentityInterface(fields=INPUTNODE_FIELDS), name="n4_buffer")

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_report", "fsl_affine"]),
        name="outputnode",
    )

    # Connect `inputnode` to `n4_buffer`
    workflow = connect_n4_nodes(inputnode, n4_buffer, workflow, INPUTNODE_FIELDS, n4_reg_flag=n4_reg_flag)

    reg_UDbold_to_UDboldtemplate = init_fsl_bbr_wf(
        bold2t1w_dof=6,
        use_bbr=False,
        bold2t1w_init="register",
        omp_nthreads=4,
        name="reg_UDbold_to_UDboldtemplate",
    )

    xfm_convert_itk_to_fsl = init_itk_to_fsl_affine_wf(name="itk_to_fsl_affine")

    # fmt: off
    workflow.connect([
        (n4_buffer, reg_UDbold_to_UDboldtemplate, [
            ("UDbold", "inputnode.in_file"),
            ("UDboldtemplate", "inputnode.t1w_brain"),
        ]),
        (reg_UDbold_to_UDboldtemplate, outputnode, [("outputnode.out_report", "out_report")]),
        (reg_UDbold_to_UDboldtemplate, xfm_convert_itk_to_fsl, [("outputnode.itk_bold_to_t1", "inputnode.itk_affine")]),
        (n4_buffer, xfm_convert_itk_to_fsl, [
            ("UDbold", "inputnode.source"),
            ("UDboldtemplate", "inputnode.reference"),
        ]),
        (xfm_convert_itk_to_fsl, outputnode, [("outputnode.fsl_affine", "fsl_affine")]),
    ])
    # fmt: on

    return workflow


def init_reg_UDboldtemplate_to_anat_wf(name: str = "reg_UDboldtemplate_to_anat_wf") -> Workflow:
    """
    UDboldtemplate: distortion corrected, mask, and temporally meaned
    anat: n4 bias field corrected, masked, regridded to bold FOV
    """

    workflow = Workflow(name=name)

    INPUTNODE_FIELDS = [
        "Dboldtemplate_run",
        "Dboldtemplate_sdc_warp",
        "masked_anat",
    ]

    inputnode = pe.Node(
        niu.IdentityInterface(fields=INPUTNODE_FIELDS),
        name="inputnode",
    )

    """
    Distortion correct and obtain a mean image of the heuristically selected
    inputnode: `Dboldtemplate_run`
    """
    Dboldtemplate_1_split = pe.Node(Split(dimension="t", out_base_name="split_bold_"), name="Dboldtemplate_1_split")
    Dboldtemplate_2_hmc = pe.Node(
        MCFLIRT(save_mats=False, save_plots=False, save_rms=False), name="Dboldtemplate_2_hmc"
    )
    Dboldtemplate_3_tmean = pe.Node(MeanImage(), name="Dboldtemplate_3_tmean")
    Dboldtemplate_4_sdc_unwarp = pe.Node(ApplyWarp(), name="Dboldtemplate_4_sdc_unwarp")
    # Brain extract the distortion corrected mean bold image
    UDboldtemplate_1_n4 = pe.Node(N4BiasFieldCorrection(), name="UDboldtemplate_1_n4")
    UDboldtemplate_2_initreg = pe.Node(FLIRT(dof=6), name="UDboldtemplate_2_initreg_anat_to_UDboldtemplate")
    UDboldtemplate_3_genmask = pe.Node(Threshold(thresh=0), name="UDboldtemplate_3_generate_mask")
    UDboldtemplate_4_applymask = pe.Node(ApplyMask(), name="UDboldtemplate_4_apply_mask")
    # Register UDboldtemplate to anat
    reg_UDboldtemplate_to_anat = init_fsl_bbr_wf(
        bold2t1w_dof=6,
        use_bbr=False,
        bold2t1w_init="register",
        omp_nthreads=4,
        name="reg_UDboldtemplate_to_anat",
    )
    xfm_convert_itk_to_fsl = init_itk_to_fsl_affine_wf(name="itk_to_fsl_affine")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_report", "fsl_affine"]),
        name="outputnode",
    )

    # fmt: off
    workflow.connect([
        (inputnode, Dboldtemplate_1_split, [("Dboldtemplate_run", "in_file")]),
        (inputnode, Dboldtemplate_2_hmc, [("Dboldtemplate_run", "in_file")]),
        (Dboldtemplate_1_split, Dboldtemplate_2_hmc, [(("out_files", _get_split_volume, 0), "ref_file")]),
        (Dboldtemplate_2_hmc, Dboldtemplate_3_tmean, [("out_file", "in_file")]),
        (inputnode, Dboldtemplate_4_sdc_unwarp, [("Dboldtemplate_sdc_warp", "field_file")]),
        (Dboldtemplate_3_tmean, Dboldtemplate_4_sdc_unwarp, [
            ("out_file", "in_file"),
            ("out_file", "ref_file"),
        ]),
        (Dboldtemplate_4_sdc_unwarp, UDboldtemplate_1_n4, [("out_file", "input_image")]),
        (inputnode, UDboldtemplate_2_initreg, [("masked_anat", "in_file")]),
        (UDboldtemplate_1_n4, UDboldtemplate_2_initreg, [("output_image","reference")]),
        (UDboldtemplate_2_initreg, UDboldtemplate_3_genmask, [("out_file", "in_file")]),
        (UDboldtemplate_1_n4, UDboldtemplate_4_applymask, [("output_image", "in_file")]),
        (UDboldtemplate_3_genmask, UDboldtemplate_4_applymask, [("out_file", "mask_file")]),
        (UDboldtemplate_4_applymask, reg_UDboldtemplate_to_anat, [("out_file", "inputnode.in_file")]),
        (inputnode, reg_UDboldtemplate_to_anat, [("masked_anat", "inputnode.t1w_brain")]),
        (reg_UDboldtemplate_to_anat, outputnode, [("outputnode.out_report", "out_report")]),
        (reg_UDboldtemplate_to_anat, xfm_convert_itk_to_fsl, [("outputnode.itk_bold_to_t1", "inputnode.itk_affine")]),
        (UDboldtemplate_4_applymask, xfm_convert_itk_to_fsl, [("out_file", "inputnode.source")]),
        (inputnode, xfm_convert_itk_to_fsl, [("masked_anat", "inputnode.reference")]),
        (xfm_convert_itk_to_fsl, outputnode, [("outputnode.fsl_affine", "fsl_affine")]),
    ])
    # fmt: on

    return workflow


def init_reg_anat_to_template_wf(template_thr: float, name: str = "reg_anat_to_template_wf") -> Workflow:
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["anat", "template"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "anat_brain",
                "init_out_report",
                "out_report",
                "fsl_warp",
            ]
        ),
        name="outputnode",
    )

    # Brain extract anatomical
    anat_1_initreg = pe.Node(
        FLIRT(
            dof=6,
            searchr_x=[-180, 180],
            searchr_y=[-180, 180],
            searchr_z=[-180, 180],
        ),
        name="anat_1_initreg",
    )
    anat_2_genmask = pe.Node(Threshold(thresh=template_thr), name="anat_1_generate_mask")
    anat_3_mask = pe.Node(ApplyMask(), name="anat_1_apply_mask")

    # Register anat to template
    reg_anat_to_template_init = pe.Node(
        FLIRTRPT(
            dof=12,
            searchr_x=[-180, 180],
            searchr_y=[-180, 180],
            searchr_z=[-180, 180],
            generate_report=True,
        ),
        name="anat_to_template_init",
    )
    reg_anat_to_template = pe.Node(RegistrationSynQuick(transform_type="sr"), name="anat_to_template")
    report_anat_to_template = pe.Node(
        ANTSApplyTransformsRPT(generate_report=True), name="generate_report_anat_to_template"
    )
    merge_transforms = pe.Node(
        Function(
            input_names=["input_1", "input_2"],
            output_names=["output_list"],
            function=_listify_two_inputs,
        ),
        name="merge_transforms",
    )
    xfm_convert_itk_to_fsl_affine = init_itk_to_fsl_affine_wf(name="itk_to_fsl_anat_to_template_affine")
    xfm_convert_itk_to_fsl_warp = init_itk_to_fsl_warp_wf(name="itk_to_fsl_anat_to_template_warp")
    merge_affines = pe.Node(ConvertXFM(concat_xfm=True), name="merge_affines")
    create_warp = pe.Node(ConvertWarp(relwarp=True), name="create_warp")

    # fmt: off
    workflow.connect([
        (inputnode, anat_1_initreg, [
            ("template", "in_file"),
            ("anat", "reference")
        ]),
        (anat_1_initreg, anat_2_genmask, [("out_file", "in_file")]),
        (inputnode, anat_3_mask, [("anat", "in_file")]),
        (anat_2_genmask, anat_3_mask, [("out_file", "mask_file")]),
        (anat_3_mask, reg_anat_to_template_init, [("out_file", "in_file")]),
        (inputnode, reg_anat_to_template_init, [("template", "reference")]),
        (reg_anat_to_template_init, reg_anat_to_template, [("out_file", "moving_image")]),
        (inputnode, reg_anat_to_template, [("template", "fixed_image")]),
        (reg_anat_to_template_init, report_anat_to_template, [("out_file", "input_image")]),
        (inputnode, report_anat_to_template, [("template", "reference_image")]),
        (reg_anat_to_template, merge_transforms, [
            ("forward_warp_field", "input_1"),
            ("out_matrix", "input_2"),
        ]),
        (merge_transforms, report_anat_to_template, [("output_list", "transforms")]),
        (reg_anat_to_template, xfm_convert_itk_to_fsl_affine, [("out_matrix", "inputnode.itk_affine")]),
        (reg_anat_to_template_init, xfm_convert_itk_to_fsl_affine, [("out_file", "inputnode.source")]),
        (inputnode, xfm_convert_itk_to_fsl_affine, [("template", "inputnode.reference")]),
        (reg_anat_to_template, xfm_convert_itk_to_fsl_warp, [("forward_warp_field", "inputnode.itk_warp")]),
        (inputnode, xfm_convert_itk_to_fsl_warp, [("template", "inputnode.reference")]),
        (reg_anat_to_template_init, merge_affines, [("out_matrix_file", "in_file")]),
        (xfm_convert_itk_to_fsl_affine, merge_affines, [("outputnode.fsl_affine", "in_file2")]),
        (merge_affines, create_warp, [("out_file", "premat")]),
        (xfm_convert_itk_to_fsl_warp, create_warp, [("outputnode.fsl_warp", "warp1")]),
        (inputnode, create_warp, [("template", "reference")]),
        (reg_anat_to_template_init, outputnode, [("out_report", "init_out_report")]),
        (report_anat_to_template, outputnode, [("out_report", "out_report")]),
        (create_warp, outputnode, [("out_file", "fsl_warp")]),
        (anat_3_mask, outputnode, [("out_file", "anat_brain")]),
    ])
    # fmt: on

    return workflow


def _get_split_volume(out_files, vol_id):
    return out_files[vol_id]


def _listify_two_inputs(input_1, input_2):
    return [input_1, input_2]
