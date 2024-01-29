import sys

sys.path.insert(1, "/opt/animalfmritools")
import os

import certifi

os.environ["REQUESTS_CA_BUNDLE"] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())
from pathlib import Path

from base_utils import setup_workflow
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from workflow_utils import (
    jsonify,
    load_json_as_dict,
    setup_buffer_nodes,
)

from animalfmritools.cli.parser import setup_parser
from animalfmritools.interfaces.evenify_nifti import EvenifyNifti
from animalfmritools.interfaces.rescale_nifti import RescaleNifti
from animalfmritools.utils.data_grabber import (
    REVERSE_PE_MAPPING,
)
from animalfmritools.workflows.anat.regrid import (
    init_regrid_anat_to_bold_wf,
    init_regrid_template_to_bold_wf,
)
from animalfmritools.workflows.bold.boldref import init_bold_ref_wf
from animalfmritools.workflows.bold.confounds import init_bold_confs_wf
from animalfmritools.workflows.bold.hmc import init_bold_hmc_wf
from animalfmritools.workflows.bold.sdc import init_bold_sdc_wf
from animalfmritools.workflows.bold.surface_mapping import init_bold_surf_wf
from animalfmritools.workflows.derivatives.base_outputs import (
    get_source_files as get_base_source_files,
)
from animalfmritools.workflows.derivatives.base_outputs import (
    init_base_preproc_derivatives_wf,
    load_base,
)
from animalfmritools.workflows.derivatives.outputs import (
    get_source_files as get_bold_source_files,
)
from animalfmritools.workflows.derivatives.outputs import (
    init_bold_preproc_derivatives_wf,
    parse_bold_path,
)
from animalfmritools.workflows.registration.apply import init_trans_bold_to_template_wf
from animalfmritools.workflows.registration.transforms import (
    init_reg_anat_to_template_wf,
    init_reg_Dbold_to_Dboldtemplate_wf,
    init_reg_Dboldtemplate_to_anat_wf,
    init_reg_UDbold_to_UDboldtemplate_wf,
    init_reg_UDboldtemplate_to_anat_wf,
)

RESCALE_FACTOR = 10  # Scale voxel sizes by 10 so that some neuroimaging tools will work for animals


def run():
    parser = setup_parser()
    args = parser.parse_args()

    force_anat = Path(args.force_anat) if isinstance(args.force_anat, str) else args.force_anat

    use_anat_to_guide = args.use_anat_to_guide
    if args.use_anat_to_guide and not args.force_anat:
        raise ValueError("--force_anat <anat_path> must be set.")

    # Subject info
    wf_manager = setup_workflow(
        args.species_id,
        args.subject_id,
        args.session_id,
        args.bids_dir,
        args.out_dir,
        args.scratch_dir,
        force_anat=force_anat,
        use_anat_to_guide=use_anat_to_guide,
        anat_contrast_type=args.anat_contrast,
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
    reorient_anat_template = pe.Node(Reorient2Std(), name="reorient_anat_template")
    rescale_anat_template = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_anat_template")
    reorient_template = pe.Node(Reorient2Std(), name="reorient_template")
    reorient_template_gm = pe.Node(Reorient2Std(), name="reorient_template_gm")
    reorient_template_wm = pe.Node(Reorient2Std(), name="reorient_template_wm")
    reorient_template_csf = pe.Node(Reorient2Std(), name="reorient_template_csf")
    rescale_template = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template")
    rescale_template_gm = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template_gm")
    rescale_template_wm = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template_wm")
    rescale_template_csf = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_template_csf")

    assert "anat_template" in buffer_nodes.anat.keys()
    # fmt: off
    wf.connect([
        (buffer_nodes.anat["anat_template"], reorient_anat_template, [("anat", "in_file")]),
        (reorient_anat_template, rescale_anat_template, [("out_file", "nifti_path")]),
        (buffer_nodes.template, reorient_template, [("template", "in_file")]),
        (reorient_template, rescale_template, [("out_file", "nifti_path")]),
        (buffer_nodes.template, reorient_template_gm, [("gm", "in_file")]),
        (reorient_template_gm, rescale_template_gm, [("out_file", "nifti_path")]),
        (buffer_nodes.template, reorient_template_wm, [("wm", "in_file")]),
        (reorient_template_wm, rescale_template_wm, [("out_file", "nifti_path")]),
        (buffer_nodes.template, reorient_template_csf, [("csf", "in_file")]),
        (reorient_template_csf, rescale_template_csf, [("out_file", "nifti_path")]),
    ])
    # fmt: on

    """
    Rescale bold runs
    """
    for run_type, runs in wf_manager.bold_runs.items():
        for ix, run_path in enumerate(runs):
            reorient_nifti = pe.Node(Reorient2Std(in_file=run_path), name=f"bold_reorient_{run_type}_{ix}")
            evenify_nifti = pe.Node(EvenifyNifti(), name=f"bold_evenify_{run_type}_{ix}")
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
            reorient_nifti = pe.Node(Reorient2Std(in_file=run_path), name=f"fmap_reorient_{run_type}_{ix}")
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
    `no_sdc` is set to True if no opposite phase-encoding gradient
    echo run is detected.
    """
    no_sdc = False

    """
    Set-up bold template (one template per PE-direction [run_type])
    - SDC unwarping will be estimated for the bold template
        - select first bold run,
        - extract first volume,
        - find a reverse PE-EPI volume,
            - look through bids fmap folder, if nothing found look for reverse PE-bold runs, and extract first volume,
        - use the reverse PE pairs to perform topup and obtain the displacement warp for the 1st volume
    """
    sdc_buffer = None
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
                session_reverse_pe_boldref = init_bold_ref_wf(name=f"session_bold_reference_opposite_pe_{run_type}")
                # fmt: off
                wf.connect([
                    (buffer_nodes.bold[reverse_run_type], session_reverse_pe_boldref, [(reverse_pe, "inputnode.bold")]),
                    (session_reverse_pe_boldref, sdc_buffer, [("outputnode.boldref", "reverse_pe")])
                ])
                # fmt: on

            except Exception:
                no_sdc = True
                print(
                    f"Warning: No reverse PE run were identified [{reverse_run_type}]\nProceed by employing linear registration from BOLD to anat."
                )

        if not no_sdc:
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
    Set-up base derivatives workflow
    """
    first_dir_type = next(iter(buffer_nodes.bold_session))
    all_dir_types = list(buffer_nodes.bold_session.keys())

    # Save bold (run-level) outputs
    base_info = load_base(args.subject_id, args.session_id)
    deriv_outputs = get_base_source_files(
        base_info,
        wf_manager.deriv_dir,
        first_dir_type,
        all_dir_types,
        args.anat_contrast,
    )
    base_deriv_wf = init_base_preproc_derivatives_wf(
        deriv_outputs, name="base_derivatives", no_sdc=no_sdc, use_anat_to_guide=use_anat_to_guide
    )

    """
    Register all `sdc_bold`  to the first run_type
    """
    if no_sdc:
        """
        Node: `buffer_nodes.bold_session_template`, output: "bold_session_template"
            - only used for regridding anatomical and template to native BOLD resolution
        """
        # fmt: off
        wf.connect([
            (sdc_buffer, buffer_nodes.bold_session_template, [("forward_pe", "bold_session_template")]),
        ])
        # fmt: on
    else:
        # fmt: off
        wf.connect([
            (buffer_nodes.bold_session[first_dir_type], buffer_nodes.bold_session_template, [("sdc_bold", "bold_session_template")]),
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
                reg_UDbold_to_UDboldtemplate = init_reg_UDbold_to_UDboldtemplate_wf(
                    n4_reg_flag=True,
                    name=f"reg_UDbold_to_UDboldtemplate_{run_type}_wf",
                )
                # fmt: off
                wf.connect([
                    (buffer_nodes.bold_session_template, reg_UDbold_to_UDboldtemplate, [("bold_session_template", "inputnode.UDboldtemplate")]),
                    (buffer_nodes.bold_session[run_type], reg_UDbold_to_UDboldtemplate, [("sdc_bold", "inputnode.UDbold")]),
                    (reg_UDbold_to_UDboldtemplate, buffer_nodes.bold_session_template_reg, [("outputnode.fsl_affine", f"bold_session_{run_type}_to_bold_session_template_reg")]),
                    (reg_UDbold_to_UDboldtemplate, base_deriv_wf, [("outputnode.out_report", f"inputnode.reg_from_UDbold{run_type}_to_UDboldtemplate")]),
                ])
                # fmt: on

    # Regrid anat
    regrid_anat_template_to_bold = init_regrid_anat_to_bold_wf(
        regrid_to_bold=False, name="regrid_anat_template_to_bold_wf"
    )

    # Regrid template
    if args.force_isotropic is not None:
        # TODO: clean up
        from nipype.interfaces.fsl import FLIRT
        from nipype.interfaces.utility import Function

        from animalfmritools.interfaces.functions import get_translation_component_from_nifti

        flirt_iso = pe.Node(FLIRT(apply_isoxfm=args.force_isotropic), name="template_force_isotropic")
        get_trans_component = pe.Node(
            interface=Function(
                input_names=["nifti_path"], output_names=["translation"], function=get_translation_component_from_nifti
            ),
            name="template_get_translation",
        )
        # fmt: off
        wf.connect([
            (reorient_template, flirt_iso, [
                ("out_file", "in_file"),
                ("out_file", "reference"),
            ]),
            (flirt_iso, get_trans_component, [("out_file", "nifti_path")]),
        ])
        # fmt: on
        args.force_isotropic *= RESCALE_FACTOR
    regrid_template_to_bold = init_regrid_template_to_bold_wf(
        force_isotropic=args.force_isotropic, name="regrid_template_to_bold_wf"
    )

    # fmt: off
    wf.connect([
        (buffer_nodes.bold_session_template, regrid_anat_template_to_bold, [("bold_session_template", "inputnode.bold")]),
        (rescale_anat_template, regrid_anat_template_to_bold, [("rescaled_path", "inputnode.anat")]),
        (buffer_nodes.bold_session_template, regrid_template_to_bold, [("bold_session_template", "inputnode.bold")]),
        (rescale_template, regrid_template_to_bold, [("rescaled_path", "inputnode.template")]),
        (rescale_template_gm, regrid_template_to_bold, [("rescaled_path", "inputnode.template_gm")]),
        (rescale_template_wm, regrid_template_to_bold, [("rescaled_path", "inputnode.template_wm")]),
        (rescale_template_csf, regrid_template_to_bold, [("rescaled_path", "inputnode.template_csf")]),
    ])
    # fmt: on

    # Register anat to template
    if args.species_id == 'marmoset':
        TEMPLATE_THRESHOLDING = 0.5
    else:
        TEMPLATE_THRESHOLDING = 5
    reg_anat_to_template = init_reg_anat_to_template_wf(TEMPLATE_THRESHOLDING, name="reg_anat_to_template_wf")

    # Register undistorted bold template to anat
    session_bold_run_input = buffer_nodes.bold_inputs[first_dir_type][0]
    reg_Dboldtemplate_to_anat = None
    reg_UDboldtemplate_to_anat = None
    if no_sdc:
        reg_Dboldtemplate_to_anat = init_reg_Dboldtemplate_to_anat_wf(name="reg_Dboldtemplate_to_anat")
        # fmt: off
        wf.connect([
            (buffer_nodes.bold[first_dir_type], reg_Dboldtemplate_to_anat, [(session_bold_run_input, "inputnode.Dboldtemplate_run")]),
            (reg_anat_to_template, reg_Dboldtemplate_to_anat, [("outputnode.anat_brain", "inputnode.masked_anat")]),
            (reg_Dboldtemplate_to_anat, base_deriv_wf, [("outputnode.out_report", "inputnode.reg_from_Dboldtemplate_to_anat")]),
        ])
        # fmt: on
    else:
        reg_UDboldtemplate_to_anat = init_reg_UDboldtemplate_to_anat_wf(name="reg_UDboldtemplate_to_anat_wf")
        # fmt: off
        wf.connect([
            (buffer_nodes.bold[first_dir_type], reg_UDboldtemplate_to_anat, [(session_bold_run_input, "inputnode.Dboldtemplate_run")]),
            (buffer_nodes.bold_session[first_dir_type], reg_UDboldtemplate_to_anat, [("sdc_warp", "inputnode.Dboldtemplate_sdc_warp")]),
            (reg_anat_to_template, reg_UDboldtemplate_to_anat, [("outputnode.anat_brain", "inputnode.masked_anat")]),
            (reg_UDboldtemplate_to_anat, base_deriv_wf, [("outputnode.out_report", "inputnode.reg_from_UDboldtemplate_to_anat")]),
        ])
        # fmt: on

    # fmt: off
    wf.connect([
        (regrid_anat_template_to_bold, reg_anat_to_template, [("outputnode.regridded_anat", "inputnode.anat")]),
        (regrid_template_to_bold, reg_anat_to_template, [("outputnode.regridded_template", "inputnode.template")]),
        (reg_anat_to_template, base_deriv_wf, [
            ("outputnode.init_out_report", "inputnode.reg_from_anat_to_template_init"),
            ("outputnode.out_report", "inputnode.reg_from_anat_to_template"),
        ]),
    ])
    # fmt: on

    # Reverse scaling of anatomical brainmask
    reverse_rescaling_anat = pe.Node(
        RescaleNifti(rescale_factor=1 / RESCALE_FACTOR),
        name="anat_brainmask_reverse_rescale",
    )

    # fmt: off
    wf.connect([
        (reg_anat_to_template, reverse_rescaling_anat, [("outputnode.anat_brain", "nifti_path")]),
        (reverse_rescaling_anat, base_deriv_wf, [("rescaled_path", "inputnode.anat_brainmask")]),
    ])
    # fmt: on

    if use_anat_to_guide:
        assert "anat_native" in buffer_nodes.anat.keys() and "anat_template" in buffer_nodes.anat.keys()
        reorient_anat_native = pe.Node(Reorient2Std(), name="reorient_anat_native")
        rescale_anat_native = pe.Node(RescaleNifti(rescale_factor=RESCALE_FACTOR), name="rescale_anat_native")
        regrid_anat_native_to_bold = init_regrid_anat_to_bold_wf(
            regrid_to_bold=False, name="regrid_anat_native_to_bold_wf"
        )
        reg_anat_native_to_template = init_reg_anat_to_template_wf(
            TEMPLATE_THRESHOLDING, name="reg_anat_native_to_template_wf"
        )
        reg_anat_native_to_anat_template = init_reg_anat_to_template_wf(
            0, skullstrip_anat=False, name="reg_anat_native_to_anat_template_wf"
        )

        # fmt: off
        wf.connect([
            (buffer_nodes.anat["anat_native"], reorient_anat_native, [("anat", "in_file")]),
            (reorient_anat_native, rescale_anat_native, [("out_file", "nifti_path")]),
            (buffer_nodes.bold_session_template, regrid_anat_native_to_bold, [("bold_session_template", "inputnode.bold")]),
            (rescale_anat_native, regrid_anat_native_to_bold, [("rescaled_path", "inputnode.anat")]),
            (regrid_anat_native_to_bold, reg_anat_native_to_template, [("outputnode.regridded_anat", "inputnode.anat")]),
            (regrid_template_to_bold, reg_anat_native_to_template, [("outputnode.regridded_template", "inputnode.template")]),
            (reg_anat_native_to_template, reg_anat_native_to_anat_template, [("outputnode.anat_brain", "inputnode.anat")]),
            (reg_anat_to_template, reg_anat_native_to_anat_template, [("outputnode.anat_brain", "inputnode.template")]),
            (reg_anat_native_to_anat_template, base_deriv_wf, [
                ("outputnode.init_out_report", "inputnode.reg_from_anatnative_to_anat_init"),
                ("outputnode.out_report", "inputnode.reg_from_anatnative_to_anat"),
            ]),
        ])
        # fmt: on

        # Link anat (not skullstripped) image to `regUDboldtemplate_to_anat/reg_Dboldtemplate_to_anat`
        if no_sdc:
            # fmt: off
            wf.disconnect([(reg_anat_to_template, reg_Dboldtemplate_to_anat, [("outputnode.anat_brain", "inputnode.masked_anat")])])
            wf.connect([(reg_anat_native_to_template, reg_Dboldtemplate_to_anat, [("outputnode.anat_brain", "inputnode.masked_anat")])])
            # wf.connect([(regrid_anat_native_to_bold, reg_Dboldtemplate_to_anat, [("outputnode.regridded_anat", "inputnode.masked_anat")])])
            # fmt: on
        else:
            # fmt: off
            wf.disconnect([(reg_anat_to_template, reg_UDboldtemplate_to_anat, [("outputnode.anat_brain", "inputnode.masked_anat")])])
            wf.connect([(reg_anat_native_to_template, reg_UDboldtemplate_to_anat, [("outputnode.anat_brain", "inputnode.masked_anat")])])
            # wf.connect([(regrid_anat_native_to_bold, reg_UDboldtemplate_to_anat, [("outputnode.regridded_anat", "inputnode.masked_anat")])])
            # fmt: on

    # Process each run
    for run_type, _bold_buffer in buffer_nodes.bold.items():
        for bold_ix, bold_input in enumerate(buffer_nodes.bold_inputs[run_type]):
            # Select bold path and load associated metadata
            prefix = f"{bold_input}_{run_type}"
            bold_path = wf_manager.bold_runs[run_type][bold_ix]
            json_path = Path(str(bold_path).replace(".nii.gz", ".json"))
            try:
                metadata = load_json_as_dict(json_path)
            except FileNotFoundError:
                print(f"{json_path} not found.\nCreating empty metadata dictionary.")
                assert args.repetition_time is not None, "Must specify --repetition-time argument."
                metadata = {}
                metadata["RepetitionTime"] = args.repetition_time
            except Exception as e:
                print(f"An unexpected error occured: {e}\nCreating empty metadata dictionary.")
                assert args.repetition_time is not None, "Must specify --repetition-time argument."
                metadata = {}
                metadata["RepetitionTime"] = args.repetition_time

            # Transform bold to template space
            boldref = init_bold_ref_wf(name=f"{prefix}_bold_reference")
            hmc = init_bold_hmc_wf(name=f"{prefix}_bold_hmc")
            reg_Dbold_to_Dboldtemplate = init_reg_Dbold_to_Dboldtemplate_wf(
                n4_reg_flag=True, name=f"{prefix}_reg_Dbold_to_Dboldtemplate"
            )
            trans_Dbold_to_template = init_trans_bold_to_template_wf(
                no_sdc=no_sdc,
                reg_quick=args.reg_quick,
                use_anat_to_guide=use_anat_to_guide,
                name=f"{prefix}_transform_Dbold_to_template",
            )

            if no_sdc:
                # fmt: off
                wf.connect([
                    (reg_Dboldtemplate_to_anat, trans_Dbold_to_template, [("outputnode.fsl_affine", "inputnode.Dboldtemplate_to_anat_aff")]),
                ])
                # fmt: on
            else:
                # fmt: off
                wf.connect([
                    (buffer_nodes.bold_session[run_type], trans_Dbold_to_template, [("sdc_warp", "inputnode.Dboldtemplate_sdc_warp")]),
                    (buffer_nodes.bold_session_template_reg, trans_Dbold_to_template, [(f"bold_session_{run_type}_to_bold_session_template_reg", "inputnode.UDbold_to_UDboldtemplate_aff")]),
                    (reg_UDboldtemplate_to_anat, trans_Dbold_to_template, [("outputnode.fsl_affine", "inputnode.UDboldtemplate_to_anat_aff")]),
                ])
                # fmt: on

            # fmt: off
            wf.connect([
                (_bold_buffer, boldref,[(bold_input, "inputnode.bold")]),
                (_bold_buffer, hmc,[(bold_input, "inputnode.bold")]),
                (boldref, hmc,[("outputnode.boldref", "inputnode.reference")]),
                (boldref, reg_Dbold_to_Dboldtemplate, [("outputnode.boldref", "inputnode.Dbold")]),
                (buffer_nodes.bold_session[run_type], reg_Dbold_to_Dboldtemplate, [("distorted_bold", "inputnode.Dboldtemplate")]),
                (_bold_buffer, trans_Dbold_to_template, [(bold_input, "inputnode.bold")]),
                (regrid_anat_template_to_bold, trans_Dbold_to_template, [("outputnode.regridded_anat", "inputnode.regridded_anat")]),
                (regrid_template_to_bold, trans_Dbold_to_template, [("outputnode.regridded_template", "inputnode.regridded_template")]),
                (hmc, trans_Dbold_to_template, [("outputnode.hmc_mats", "inputnode.Dbold_hmc_affs")]),
                (reg_Dbold_to_Dboldtemplate, trans_Dbold_to_template, [("outputnode.fsl_affine", "inputnode.Dbold_to_Dboldtemplate_aff")]),
                (reg_anat_to_template, trans_Dbold_to_template, [("outputnode.fsl_warp", "inputnode.anat_to_template_warp")]),
            ])
            # fmt: on

            # Estimate confounds
            bold_confs_wf = init_bold_confs_wf(
                mem_gb=8,
                metadata=metadata,
                freesurfer=True,
                regressors_all_comps=False,
                regressors_dvars_th=1.5,
                regressors_fd_th=0.5,
                name=f"{prefix}_bold_confounds",
            )
            bold_confs_wf.inputs.inputnode.skip_vols = 0
            bold_confs_wf.inputs.inputnode.t1_bold_xform = "identity"
            # fmt: off
            wf.connect([
                (trans_Dbold_to_template, bold_confs_wf, [("outputnode.bold_template_space", "inputnode.bold")]),
                (regrid_template_to_bold, bold_confs_wf, [
                    ("outputnode.regridded_template_mask", "inputnode.bold_mask"),
                    ("outputnode.regridded_template_mask", "inputnode.t1w_mask"),
                    ("outputnode.regridded_template_tpms", "inputnode.t1w_tpms"),
                ]),
                (hmc, bold_confs_wf, [
                    ("outputnode.movpar_file", "inputnode.movpar_file"),
                    ("outputnode.rmsd_file", "inputnode.rmsd_file")
                ]),
            ])
            # fmt: on

            # Reverse scaling of BOLD
            reverse_rescaling = pe.Node(
                RescaleNifti(rescale_factor=1 / RESCALE_FACTOR),
                name=f"{prefix}_bold_reverse_rescale",
            )
            if args.force_isotropic is not None:
                # fmt: off
                wf.connect([
                    (get_trans_component, reverse_rescaling, [("translation", "force_translation")]),
                ])
                # fmt: on

            # Surface projection of BOLD
            surface_projection = init_bold_surf_wf(metadata, name=f"{prefix}_bold_surface_mapping")
            # fmt: off
            wf.connect([
                (reverse_rescaling, surface_projection, [("rescaled_path", "inputnode.bold_nifti")]),
                (buffer_nodes.surfaces, surface_projection, [
                    ("lh_midthickness", "inputnode.lh_midthickness"),
                    ("rh_midthickness", "inputnode.rh_midthickness"),
                    ("lh_white", "inputnode.lh_white"),
                    ("rh_white", "inputnode.rh_white"),
                    ("lh_pial", "inputnode.lh_pial"),
                    ("rh_pial", "inputnode.rh_pial"),
                    ("lh_cortex", "inputnode.lh_cortex"),
                    ("rh_cortex", "inputnode.rh_cortex"),
                ]),
            ])
            # fmt: on

            # Save bold (run-level) outputs
            bold_info = parse_bold_path(bold_path)
            deriv_outputs = get_bold_source_files(bold_info, wf_manager.deriv_dir)
            bold_deriv_wf = init_bold_preproc_derivatives_wf(
                deriv_outputs,
                name=f"{prefix}_bold_derivatives",
            )

            # fmt: off
            wf.connect([
                (trans_Dbold_to_template, reverse_rescaling, [("outputnode.bold_template_space", "nifti_path")]),
                (reverse_rescaling, bold_deriv_wf, [("rescaled_path", "inputnode.bold_preproc")]),
                (surface_projection, bold_deriv_wf, [("outputnode.bold_dtseries", "inputnode.bold_preproc_dtseries")]),
                (bold_confs_wf, bold_deriv_wf, [
                    (("outputnode.confounds_metadata", jsonify), "inputnode.bold_confounds_metadata"),
                    ("outputnode.confounds_file", "inputnode.bold_confounds"),
                    ("outputnode.rois_plot", "inputnode.bold_roi_svg"),
                ]),
                (reg_Dbold_to_Dboldtemplate, bold_deriv_wf, [("outputnode.out_report", "inputnode.reg_from_Dbold_to_Dboldtemplate")]),
            ])
            # fmt: on

            if use_anat_to_guide:
                # fmt: off
                wf.disconnect([(regrid_anat_template_to_bold, trans_Dbold_to_template, [("outputnode.regridded_anat", "inputnode.regridded_anat")])])
                wf.connect([
                    (regrid_anat_native_to_bold, trans_Dbold_to_template, [("outputnode.regridded_anat", "inputnode.regridded_anat")]),
                    (reg_anat_native_to_anat_template, trans_Dbold_to_template, [("outputnode.fsl_warp", "inputnode.anat_native_to_anat_secondary_warp")]),
                ])
                # fmt: on

    wf.run()


if __name__ == "__main__":
    run()
