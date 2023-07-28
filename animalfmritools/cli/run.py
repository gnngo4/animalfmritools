import sys
sys.path.insert(1, "/opt/animalfmritools")

import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = os.path.join(
    os.path.dirname(sys.argv[0]), certifi.where()
)

from pathlib import Path

from animalfmritools.utils.data_grabber import (
    REVERSE_PE_MAPPING,
    PE_DIR_FLIP,
)
from animalfmritools.cli.parser import setup_parser
from animalfmritools.interfaces.apply_bold_to_anat import ApplyBoldToAnat
from animalfmritools.interfaces.rescale_nifti import RescaleNifti
from animalfmritools.interfaces.copy_aff_hdr_nifti import CopyAffineHeaderInfo
from animalfmritools.interfaces.flip_nifti import FlipNifti
from animalfmritools.interfaces.evenify_nifti import EvenifyNifti
from animalfmritools.workflows.bold.boldref import init_bold_ref_wf
from animalfmritools.workflows.bold.confounds import init_bold_confs_wf
from animalfmritools.workflows.bold.sdc import init_bold_sdc_wf
from animalfmritools.workflows.derivatives.outputs import (
    parse_bold_path,
    get_source_files,
    init_bold_preproc_derivatives_wf,
)
from animalfmritools.workflows.registration.utils import init_itk_to_fsl_affine_wf
from animalfmritools.workflows.registration.utils import init_itk_to_fsl_warp_wf
from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from nipype.interfaces.fsl import MCFLIRT, FLIRT, ApplyWarp, ConvertXFM, ConvertWarp
from nipype.interfaces.fsl.maths import Threshold, ApplyMask, MeanImage, UnaryMaths
from nipype.interfaces.fsl.utils import Split, Reorient2Std
from nipype.interfaces.ants import (
    N4BiasFieldCorrection,
    RegistrationSynQuick,
    ApplyTransforms,
)
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nibabel import GenerateSamplingReference
from niworkflows.interfaces.confounds import NormalizeMotionParams
from fmriprep.workflows.bold.registration import init_fsl_bbr_wf

from workflow_utils import (
    jsonify,
    load_json_as_dict,
    pick_rel,
    listify_three_inputs,
    get_split_volume,
)
from base_utils import setup_workflow
from workflow_utils import setup_buffer_nodes

RESCALE_FACTOR = (
    10  # Scale voxel sizes by 10 so that some neuroimaging tools will work for animals
)
TEMPLATE_THRESHOLDING = 5


def run():
    parser = setup_parser()
    args = parser.parse_args()

    # Subject info
    wf_manager = setup_workflow(
        args.subject_id, 
        args.session_id, 
        args.bids_dir, 
        args.out_dir, 
        args.scratch_dir
    )
    
    # Instantiate workflow
    wf = Workflow(
        name=f"animalfmritools_sub-{wf_manager.sub_id}_ses-{wf_manager.ses_id}",
        base_dir=wf_manager.scratch_dir,
    )

    # Set-up buffer nodes
    buffer_nodes = setup_buffer_nodes(wf_manager)

    """
    Rescale anat and template
    """
    reorient_anat = pe.Node(
        Reorient2Std(), name=f"reorient_anat"
    )
    rescale_anat = pe.Node(
        RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_anat"
    )
    n4_anat = pe.Node(N4BiasFieldCorrection(), name="n4_anat")
    rescale_template = pe.Node(
        RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template"
    )
    rescale_template_gm = pe.Node(
        RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template_gm"
    )
    rescale_template_wm = pe.Node(
        RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template_wm"
    )
    rescale_template_csf = pe.Node(
        RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template_csf"
    )

    # fmt: off
    wf.connect([
        (buffer_nodes.anat, reorient_anat, [("t2w", "in_file")]),
        (reorient_anat, rescale_anat, [("out_file", "nifti_path")]),
        (rescale_anat, n4_anat, [("rescaled_path", "input_image")]),
        (buffer_nodes.template, rescale_template, [("template", "nifti_path")]),
        (buffer_nodes.template, rescale_template_gm, [("gm", "nifti_path")]),
        (buffer_nodes.template, rescale_template_wm, [("wm", "nifti_path")]),
        (buffer_nodes.template, rescale_template_csf, [("csf", "nifti_path")]),
    ])
    # fmt: on

    """
    Rescale bold runs
    """
    for run_type, runs in wf_manager.bold_runs.items():
        for ix, run_path in enumerate(runs):
            reorient_nifti = pe.Node(
                Reorient2Std(in_file=run_path), name=f"bold_reorient_{run_type}_{ix}"
            )
            evenify_nifti = pe.Node(
                EvenifyNifti(), name=f"bold_evenify_{run_type}_{ix}"
            )
            rescale_nifti = pe.Node(
                RescaleNifti(rescale_factor=RESCALE_FACTOR),
                name=f"bold_rescale_{run_type}_{ix}",
            )
            # fmt: off
            wf.connect([
                (reorient_nifti, evenify_nifti, [("out_file", "nifti_path")]),
                (evenify_nifti, rescale_nifti, [("out_path", "nifti_path")]),
                (rescale_nifti, buffer_nodes.bold[run_type], [("rescaled_path", buffer_nodes.bold_inputs[run_type][ix])]),
            ])
            # fmt: on

    """
    Rescale fmap runs
    """
    for run_type, runs in wf_manager.fmap_runs.items():
        for ix, run_path in enumerate(runs):
            reorient_nifti = pe.Node(
                Reorient2Std(in_file=run_path), name=f"fmap_reorient_{run_type}_{ix}"
            )
            evenify_nifti = pe.Node(
                EvenifyNifti(),
                name=f"fmap_evenify_{run_type}_{ix}",
            )
            rescale_nifti = pe.Node(
                RescaleNifti(rescale_factor=RESCALE_FACTOR),
                name=f"fmap_rescale_{run_type}_{ix}",
            )
            # fmt: off
            wf.connect([
                (reorient_nifti, evenify_nifti, [("out_file", "nifti_path")]),
                (evenify_nifti, rescale_nifti, [("out_path", "nifti_path")]),
                (rescale_nifti, buffer_nodes.fmap[run_type], [("rescaled_path", buffer_nodes.fmap_inputs[run_type][ix])]),
            ])
            # fmt: on

    """
    Set-up bold template (one template per PE-direction [run_type])
    - SDC unwarping will be estimated for the bold template
        - select first bold run,
        - extract first volume,
        - find a reverse PE-EPI volume,
            - look through bids fmap folder, if nothing found look for reverse PE-bold runs, and extract first volume,
        - use the reverse PE pairs to perform topup and obtain the displacement warp for the 1st volume
    """
    for run_type, runs in buffer_nodes.bold_inputs.items():
        sdc_buffer = pe.Node(
            niu.IdentityInterface(["forward_pe', 'reverse_pe"]),
            name=f"sdc_buffer_{run_type}",
        )
        # Get reverse PE direction
        reverse_run_type = REVERSE_PE_MAPPING[run_type]
        # Get boldref
        boldref_ses = runs[0]
        forward_pe_metadata = wf_manager.bold_runs[run_type][0]
        # Extract boldref
        session_boldref = init_bold_ref_wf(name=f"session_bold_reference_{run_type}")
        n4_session_boldref = pe.Node(N4BiasFieldCorrection(), name=f"n4_session_bold_reference_{run_type}")
        # fmt: off
        wf.connect([
            (buffer_nodes.bold[run_type], session_boldref, [(boldref_ses, "inputnode.bold")]),
            (session_boldref, sdc_buffer, [("outputnode.boldref", "forward_pe")]),
            (session_boldref, n4_session_boldref, [("outputnode.boldref", "input_image")]),
            (n4_session_boldref, buffer_nodes.bold_session[run_type], [("output_image", "distorted_bold")]),
        ])
        # fmt: on
        # Get boldref (reverse PE direction)
        try:
            reverse_pe = buffer_nodes.fmap_inputs[reverse_run_type][0]
            reverse_pe_metadata = wf_manager.fmap_runs[reverse_run_type][0]
            # fmt: off
            wf.connect([
                (buffer_nodes.fmap[reverse_run_type], sdc_buffer, [(reverse_pe, "reverse_pe")])
            ])
            # fmt: on

        except Exception:
            try:
                reverse_pe = buffer_nodes.bold_inputs[reverse_run_type][0]
                reverse_pe_metadata = wf_manager.bold_runs[reverse_run_type][0]
                session_reverse_pe_boldref = init_bold_ref_wf(
                    name=f"session_bold_reference_opposite_pe_{run_type}"
                )
                # fmt: off
                wf.connect([
                    (buffer_nodes.bold[reverse_run_type], session_reverse_pe_boldref, [(reverse_pe, "inputnode.bold")]),
                    (session_reverse_pe_boldref, sdc_buffer, [("outputnode.boldref", "reverse_pe")])
                ])
                # fmt: on

            except Exception:
                raise ValueError(
                    f"A reverse PE run could not be found [{reverse_run_type}]."
                )

        session_sdc = init_bold_sdc_wf(
            forward_pe_metadata,
            reverse_pe_metadata,
            name=f"session_bold_sdc_{run_type}",
        )
        # fmt: off
        wf.connect([
            (sdc_buffer, session_sdc, [("forward_pe", "inputnode.forward_pe")]),
            (sdc_buffer, session_sdc, [("reverse_pe", "inputnode.reverse_pe")]),
            (session_sdc, buffer_nodes.bold_session[run_type], [
                ("outputnode.sdc_warp", "sdc_warp"),
                ("outputnode.sdc_bold", "sdc_bold"),
                ("outputnode.sdc_affine", "sdc_affine"),
            ])
        ])
        # fmt: on

    """
    Register all `sdc_bold`  to the first run_type
    """
    first_key = next(iter(buffer_nodes.bold_session))

    # fmt: off
    wf.connect([
        (buffer_nodes.bold_session[first_key], buffer_nodes.bold_session_template, [("sdc_bold", "bold_session_template")]),
    ])
    # fmt: on
    for session_ix, run_type in enumerate(buffer_nodes.bold_session.keys()):
        if session_ix == 0:
            # fmt: off
            wf.connect([
                (buffer_nodes.bold_session[run_type], buffer_nodes.bold_session_template_reg, [("sdc_affine", f"bold_session_{run_type}_to_bold_session_template_reg")])
            ])
            # fmt: on
        else:
            reg_bold_to_boldtemplate = init_fsl_bbr_wf(
                bold2t1w_dof=6,
                use_bbr=False,
                bold2t1w_init="register",
                omp_nthreads=4,
                name=f"reg_sdc-bold_to_sdc-boldtemplate_{run_type}",
            )
            xfm_convert_itk_to_fsl = init_itk_to_fsl_affine_wf(
                name=f"itk_to_fsl_bold_to_boldtemplate_{run_type}"
            )
            # fmt: off
            wf.connect([
                (buffer_nodes.bold_session_template, reg_bold_to_boldtemplate, [("bold_session_template", "inputnode.t1w_brain")]),
                (buffer_nodes.bold_session[run_type], reg_bold_to_boldtemplate, [("sdc_bold", "inputnode.in_file")]),
                (reg_bold_to_boldtemplate, xfm_convert_itk_to_fsl, [("outputnode.itk_bold_to_t1", "inputnode.itk_affine")]),
                (buffer_nodes.bold_session[run_type], xfm_convert_itk_to_fsl, [("sdc_bold", "inputnode.source")]),
                (buffer_nodes.bold_session_template, xfm_convert_itk_to_fsl, [("bold_session_template", "inputnode.reference")]),
                (xfm_convert_itk_to_fsl, buffer_nodes.bold_session_template_reg, [("outputnode.fsl_affine", f"bold_session_{run_type}_to_bold_session_template_reg")])
            ])
            # fmt: on

    # Regrid anat and template
    regrid_template = pe.Node(GenerateSamplingReference(), name="regrid_template")
    binarize_template = pe.Node(UnaryMaths(operation="bin"), name="binarize_template")
    regrid_template_gm = pe.Node(
        ApplyTransforms(transforms="identity"), name="regrid_template_gm"
    )
    regrid_template_wm = pe.Node(
        ApplyTransforms(transforms="identity"), name="regrid_template_wm"
    )
    regrid_template_csf = pe.Node(
        ApplyTransforms(transforms="identity"), name="regrid_template_csf"
    )
    regrid_t2w = pe.Node(GenerateSamplingReference(), name="regrid_t2w")
    merge_template_tpms = pe.Node(
        Function(
            input_names=["input_1", "input_2", "input_3"],
            output_names=["output_list"],
            function=listify_three_inputs,
        ),
        name="merge_template_tpms",
    )

    # Register session bold template to T2w
    initreg_boldtemplate_to_t2w_boldref = pe.Node(
        Split(dimension="t", out_base_name="split_bold_"),
        name="initreg_boldtemplate_to_t2w_boldref",
    )
    initreg_boldtemplate_to_t2w_hmc = pe.Node(
        MCFLIRT(save_mats=False, save_plots=False, save_rms=False),
        name="initreg_boldtemplate_to_t2w_hmc",
    )
    initreg_boldtemplate_to_t2w_tmean = pe.Node(
        MeanImage(), name="initreg_boldtemplate_to_t2w_tmean"
    )
    initreg_boldtemplate_to_t2w_n4 = pe.Node(
        N4BiasFieldCorrection(), name="initreg_boldtemplate_to_t2w_n4"
    )
    initreg_boldtemplate_to_t2w_sdc = pe.Node(
        ApplyWarp(), name="initreg_boldtemplate_to_t2w_warp"
    )
    reg_boldtemplate_to_t2w = init_fsl_bbr_wf(
        bold2t1w_dof=6,
        use_bbr=False,
        bold2t1w_init="register",
        omp_nthreads=4,
        name=f"reg_sdc-boldtemplate_to_t2w_{first_key}",
    )
    xfm_convert_itk_to_fsl_boldtemplate_to_t2w = init_itk_to_fsl_affine_wf(
        name="itk_to_fsl_boldtemplate_to_t2w"
    )

    session_bold_run_input = buffer_nodes.bold_inputs[first_key][0]
    # fmt: off
    wf.connect([
        (buffer_nodes.bold[first_key], initreg_boldtemplate_to_t2w_hmc, [(session_bold_run_input, "in_file")]),
        (buffer_nodes.bold[first_key], initreg_boldtemplate_to_t2w_boldref, [(session_bold_run_input, "in_file")]),
        (initreg_boldtemplate_to_t2w_boldref, initreg_boldtemplate_to_t2w_hmc, [(("out_files", get_split_volume, 0), "ref_file")]),
        (initreg_boldtemplate_to_t2w_hmc, initreg_boldtemplate_to_t2w_tmean, [("out_file", "in_file")]),
        (initreg_boldtemplate_to_t2w_tmean, initreg_boldtemplate_to_t2w_n4, [("out_file", "input_image")]),
        (initreg_boldtemplate_to_t2w_n4, initreg_boldtemplate_to_t2w_sdc, [
            ("output_image", "in_file"),
            ("output_image", "ref_file"),
        ]),
        (buffer_nodes.bold_session[first_key], initreg_boldtemplate_to_t2w_sdc, [("sdc_warp", "field_file")]),
        (n4_anat, regrid_t2w, [("output_image", "fixed_image")]),
        (buffer_nodes.bold_session_template, regrid_t2w, [("bold_session_template", "moving_image")]),
        (initreg_boldtemplate_to_t2w_sdc, reg_boldtemplate_to_t2w, [("out_file", "inputnode.in_file")]),
        (regrid_t2w, reg_boldtemplate_to_t2w, [("out_file", "inputnode.t1w_brain")]),
        (reg_boldtemplate_to_t2w, xfm_convert_itk_to_fsl_boldtemplate_to_t2w, [("outputnode.itk_bold_to_t1", "inputnode.itk_affine")]),
        (initreg_boldtemplate_to_t2w_sdc, xfm_convert_itk_to_fsl_boldtemplate_to_t2w, [("out_file", "inputnode.source")]),
        (regrid_t2w, xfm_convert_itk_to_fsl_boldtemplate_to_t2w, [("out_file", "inputnode.reference")]),
    ])
    # fmt: on

    # Register T2w to template
    mask_t2w_initreg = pe.Node(
        FLIRT(
            dof=6,
            searchr_x=[-180, 180],
            searchr_y=[-180, 180],
            searchr_z=[-180, 180],
        ),
        name="mask_t2w_initreg_template_to_t2w",
    )
    mask_t2w_genmask = pe.Node(Threshold(thresh=TEMPLATE_THRESHOLDING), name="mask_t2w_generate_mask")
    mask_t2w = pe.Node(ApplyMask(), name="mask_t2w_apply_mask")
    initreg_t2w_to_template = pe.Node(
        FLIRT(
            dof=12,
            searchr_x=[-180, 180],
            searchr_y=[-180, 180],
            searchr_z=[-180, 180],
        ),
        name="reg_affine_t2w_to_template",
    )
    reg_t2w_to_template = pe.Node(
        RegistrationSynQuick(transform_type="sr"), name="reg_t2w_to_template"
    )
    xfm_convert_itk_to_fsl_t2w_to_template_affine = init_itk_to_fsl_affine_wf(
        name="itk_to_fsl_t2w_to_template_affine"
    )
    xfm_convert_itk_to_fsl_t2w_to_template_warp = init_itk_to_fsl_warp_wf(
        name="itk_to_fsl_t2w_to_template_warp"
    )
    init_xfm_t2w_to_template_affine = pe.Node(
        ConvertXFM(concat_xfm=True), name="itk_to_fsl_t2w_to_template_mergeaffines"
    )
    init_xfm_t2w_to_template_warp = pe.Node(
        ConvertWarp(relwarp=True), name="itk_to_fsl_t2w_to_template_createwarp"
    )
    apply_t2w_to_template = pe.Node(ApplyWarp(), name="trans_t2w_to_template")
    # fmt: off
    wf.connect([
        (rescale_template, regrid_template, [("rescaled_path", "fixed_image")]),
        (buffer_nodes.bold_session_template, regrid_template, [("bold_session_template", "moving_image")]),
        (regrid_template, mask_t2w_initreg, [("out_file", "in_file")]),
        (regrid_t2w, mask_t2w_initreg, [("out_file", "reference")]),
        (mask_t2w_initreg, mask_t2w_genmask, [("out_file", "in_file")]),
        (mask_t2w_genmask, mask_t2w, [("out_file", "mask_file")]),
        (regrid_t2w, mask_t2w, [("out_file", "in_file")]),
        (mask_t2w, initreg_t2w_to_template, [("out_file", "in_file")]),
        (regrid_template, initreg_t2w_to_template, [("out_file", "reference")]),
        (initreg_t2w_to_template, reg_t2w_to_template, [("out_file", "moving_image")]),
        (regrid_template, reg_t2w_to_template, [("out_file", "fixed_image")]),
        (reg_t2w_to_template, xfm_convert_itk_to_fsl_t2w_to_template_affine, [("out_matrix", "inputnode.itk_affine")]),
        (initreg_t2w_to_template, xfm_convert_itk_to_fsl_t2w_to_template_affine, [("out_file", "inputnode.source")]),
        (regrid_template, xfm_convert_itk_to_fsl_t2w_to_template_affine, [("out_file", "inputnode.reference")]),
        (reg_t2w_to_template, xfm_convert_itk_to_fsl_t2w_to_template_warp, [("forward_warp_field", "inputnode.itk_warp")]),
        (regrid_template, xfm_convert_itk_to_fsl_t2w_to_template_warp, [("out_file", "inputnode.reference")]),
        (initreg_t2w_to_template, init_xfm_t2w_to_template_affine, [("out_matrix_file", "in_file")]),
        (xfm_convert_itk_to_fsl_t2w_to_template_affine, init_xfm_t2w_to_template_affine, [("outputnode.fsl_affine", "in_file2")]),    
        (init_xfm_t2w_to_template_affine, init_xfm_t2w_to_template_warp, [("out_file", "premat")]),
        (xfm_convert_itk_to_fsl_t2w_to_template_warp, init_xfm_t2w_to_template_warp, [("outputnode.fsl_warp", "warp1")]),
        (regrid_template, init_xfm_t2w_to_template_warp, [("out_file", "reference")]),
        (regrid_template, apply_t2w_to_template, [("out_file", "ref_file")]),
        (mask_t2w, apply_t2w_to_template, [("out_file", "in_file")]),
        (init_xfm_t2w_to_template_warp, apply_t2w_to_template, [("out_file", "field_file")]),
        (regrid_template, binarize_template, [("out_file", "in_file")]),
        (regrid_template, regrid_template_gm, [("out_file", "reference_image")]),
        (rescale_template_gm, regrid_template_gm, [("rescaled_path", "input_image")]),
        (regrid_template, regrid_template_wm, [("out_file", "reference_image")]),
        (rescale_template_wm, regrid_template_wm, [("rescaled_path", "input_image")]),
        (regrid_template, regrid_template_csf, [("out_file", "reference_image")]),
        (rescale_template_csf, regrid_template_csf, [("rescaled_path", "input_image")]),
        (regrid_template_gm, merge_template_tpms, [("output_image", "input_1")]),
        (regrid_template_wm, merge_template_tpms, [("output_image", "input_2")]),
        (regrid_template_csf, merge_template_tpms, [("output_image", "input_3")]),
    ])
    # fmt: on

    # Process each run
    for run_type, _bold_buffer in buffer_nodes.bold.items():
        for bold_ix, bold_input in enumerate(buffer_nodes.bold_inputs[run_type]):
            bold_path = wf_manager.bold_runs[run_type][bold_ix]
            metadata = load_json_as_dict(
                Path(str(bold_path).replace(".nii.gz", ".json"))
            )

            boldref = init_bold_ref_wf(name=f"{bold_input}_reference_{run_type}")
            n4_boldref = pe.Node(N4BiasFieldCorrection(), name=f"{bold_input}_n4_reference_{run_type}")
            hmc = pe.Node(
                MCFLIRT(save_mats=True, save_plots=True, save_rms=True),
                name=f"{bold_input}_hmc_{run_type}",
            )
            normalize_motion = pe.Node(
                NormalizeMotionParams(format="FSL"),
                name=f"{bold_input}_normalize_motion_{run_type}",
            )
            reg_bold_to_boldtemplate = init_fsl_bbr_wf(
                bold2t1w_dof=6,
                use_bbr=False,
                bold2t1w_init="register",
                omp_nthreads=4,
                name=f"reg_{bold_input}_to_nosdc-boldtemplate_{run_type}",
            )
            xfm_convert_itk_to_fsl = init_itk_to_fsl_affine_wf(
                name=f"itk_to_fsl_{bold_input}_to_nosdc-boldtemplate_{run_type}"
            )

            # fmt: off
            wf.connect([
                (_bold_buffer, boldref,[(bold_input, "inputnode.bold")]),
                (_bold_buffer, hmc,[(bold_input, "in_file")]),
                (boldref, hmc,[("outputnode.boldref", "ref_file")]),
                (hmc, normalize_motion,[("par_file", "in_file")]), # NOTE: par_file are rescaled by a factor of 10 due to the `rescale_bold` step
                (boldref, n4_boldref, [("outputnode.boldref", "input_image")]),
                (n4_boldref, reg_bold_to_boldtemplate, [("output_image", "inputnode.in_file")]),
                (buffer_nodes.bold_session[run_type], reg_bold_to_boldtemplate, [("distorted_bold", "inputnode.t1w_brain")]),
                (reg_bold_to_boldtemplate, xfm_convert_itk_to_fsl, [("outputnode.itk_bold_to_t1", "inputnode.itk_affine")]),
                (n4_boldref, xfm_convert_itk_to_fsl, [("output_image", "inputnode.source")]),
                (buffer_nodes.bold_session[run_type], xfm_convert_itk_to_fsl, [("distorted_bold", "inputnode.reference")]),
            ])
            # fmt: on

            # Merge transforms
            # 1) HMC regs to boldref [run | nosdc] `hmc` [inputs: `mat_file`]
            # 2) reg boldref [run | nosdc] to boldref [session | nosdc] `xfm_convert_itk_to_fsl` [inputs: `fsl_affine`]
            # 3) sdc warp [session] `session_bold_buffer[run_type]` [inputs: `sdc_warp`]
            # 4) reg boldref [session | sdc] to boldref [MAIN - runtype | session | sdc] `session_template_reg_buffer` [inputs: `aff_bold_to_boldtemplate_{run_type}`]
            # 5) reg boldref [MAIN - runtype | session | sdc] to T2w `xfm_convert_itk_to_fsl_boldtemplate_to_t2w` [inputs: `fsl_affine`]
            # 6) reg T2w to ABI template
            merge_4_5 = pe.Node(
                ConvertXFM(concat_xfm=True),
                name=f"merge_xfms_4-5_{bold_input}_{run_type}",
            )
            merge_2_5 = pe.Node(
                ConvertWarp(output_type="NIFTI_GZ", relwarp=True),
                name=f"merge_xfms_2-5_{bold_input}_{run_type}",
            )
            merge_2_6 = pe.Node(
                ConvertWarp(output_type="NIFTI_GZ", relwarp=True),
                name=f"merge_xfms_2-6_{bold_input}_{run_type}",
            )

            # Apply
            apply_bold_to_template = pe.Node(
                ApplyBoldToAnat(debug=args.reg_quick),
                name=f"trans_{bold_input}_to_template_{run_type}",
            )

            # fmt: off
            wf.connect([
                (buffer_nodes.bold_session_template_reg, merge_4_5, [(f"bold_session_{run_type}_to_bold_session_template_reg", "in_file")]),
                (xfm_convert_itk_to_fsl_boldtemplate_to_t2w, merge_4_5, [("outputnode.fsl_affine", "in_file2")]),
                (xfm_convert_itk_to_fsl, merge_2_5, [("outputnode.fsl_affine", "premat")]),
                (buffer_nodes.bold_session[run_type], merge_2_5, [("sdc_warp", "warp1")]),
                (merge_4_5, merge_2_5, [("out_file", "postmat")]),
                (regrid_t2w, merge_2_5, [("out_file", "reference")]),
                (merge_2_5, merge_2_6, [("out_file", "warp1")]),
                (init_xfm_t2w_to_template_warp, merge_2_6, [("out_file", "warp2")]),
                (regrid_template, merge_2_6, [("out_file", "reference")]),
                (hmc, apply_bold_to_template, [("mat_file", "hmc_mats")]),
                (_bold_buffer, apply_bold_to_template,[(bold_input, "bold_path")]),
                (merge_2_6, apply_bold_to_template,[("out_file", "bold_to_anat_warp")]),
                (regrid_template, apply_bold_to_template, [("out_file", "anat_resampled")]),
            ])
            # fmt: on

            bold_confs_wf = init_bold_confs_wf(
                mem_gb=8,
                metadata=metadata,
                freesurfer=True,
                regressors_all_comps=False,
                regressors_dvars_th=1.5,
                regressors_fd_th=0.5,
                name=f"confounds_{bold_input}_{run_type}_wf",
            )
            bold_confs_wf.inputs.inputnode.skip_vols = 0
            bold_confs_wf.inputs.inputnode.t1_bold_xform = "identity"
            # connect
            # fmt: off
            wf.connect([
                (apply_bold_to_template, bold_confs_wf, [("t1_bold_path", "inputnode.bold")]),
                (binarize_template, bold_confs_wf, [("out_file", "inputnode.bold_mask")]),
                (binarize_template, bold_confs_wf, [("out_file", "inputnode.t1w_mask")]),
                (normalize_motion, bold_confs_wf, [("out_file", "inputnode.movpar_file")]),
                (hmc, bold_confs_wf, [(("rms_files", pick_rel), "inputnode.rmsd_file")]),
                (merge_template_tpms, bold_confs_wf, [("output_list", "inputnode.t1w_tpms")]),
            ])
            # fmt: on

            deriv_outputs = get_source_files(
                parse_bold_path(bold_path), wf_manager.deriv_dir
            )
            bold_deriv_wf = init_bold_preproc_derivatives_wf(
                deriv_outputs,
                name=f"derivatives_{bold_input}_{run_type}_wf",
            )
            reverse_rescaling = pe.Node(
                RescaleNifti(rescale_factor=1 / RESCALE_FACTOR),
                name=f"reverse_rescale_{bold_input}_{run_type}",
            )
            # fmt: off
            wf.connect([
                (apply_bold_to_template, reverse_rescaling, [("t1_bold_path", "nifti_path")]),
                (reverse_rescaling, bold_deriv_wf, [("rescaled_path", "inputnode.bold_preproc")]),
                (bold_confs_wf, bold_deriv_wf, [
                    (("outputnode.confounds_metadata", jsonify), "inputnode.bold_confounds_metadata"),
                    ("outputnode.confounds_file", "inputnode.bold_confounds"),
                    ("outputnode.rois_plot", "inputnode.bold_roi_svg")
                ]),
                (reg_bold_to_boldtemplate, bold_deriv_wf, [("outputnode.out_report", "inputnode.reg_from_Dbold_to_Dboldref")]),
            ])
            # fmt: on

    wf.run()


if __name__ == "__main__":
    run()
