import argparse


def bold_to_anat_dof(arg: str) -> int:
    """Validate and convert the degree of freedom argument for BOLD to anatomical registration.

    Args:
        arg (str): Input argument.

    Returns:
        int: Validated degree of freedom value.

    Raises:
        argparse.ArgumentTypeError: If the input value is not one of [6, 7, 9, 12].
    """
    arg_int = int(arg)
    if arg_int not in {6, 7, 9, 12}:
        raise argparse.ArgumentTypeError("Invalid option. Choose: [6, 7, 9, 12]")
    return arg_int


def setup_parser() -> argparse.ArgumentParser:
    """Set up Python's ArgumentParser for animalfmritools.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject_id",
        required=True,
        type=str,
        help="subject ID in the BIDS directory.",
    )

    parser.add_argument(
        "--session_id",
        required=True,
        type=str,
        help="session ID in the BIDS directory.",
    )

    parser.add_argument("--bids_dir", required=True, type=str, help="BIDS directory.")

    parser.add_argument("--out_dir", required=True, type=str, help="output directory.")

    parser.add_argument(
        "--species_id",
        default="mouse",
        type=str,
        help="species ID - affects which template is chosen.\nOnly mouse, rat, and marmoset are supported.",
    )

    parser.add_argument(
        "--repetition_time",
        default=None,
        type=float,
        help="Manual input repetition time (TR).\nMust specify if json file is missing.\nEnabling this feature will override repetition time found in json file.",
    )

    parser.add_argument(
        "--force_isotropic", default=None, type=float, help="Force isotropic resampling to a specified resolution."
    )

    parser.add_argument(
        "--force_anat",
        default=None,
        type=str,
        help="Manual input anatomical image [.nii.gz].\nMust specify if anat is missing",
    )

    parser.add_argument(
        "--use_anat_to_guide",
        action='store_true',
        help="Use manual inputted anatomical image to guide native anat-to-template registration\nMust set --force_anat flag",
    )

    parser.add_argument(
        "--anat_contrast",
        default="T2w",
        type=str,
        help="Manual input of anatomical contrast to search for (default: T2w).",
    )

    parser.add_argument(
        "--scratch_dir",
        default="/tmp",
        type=str,
        help="workflow output directory.",
    )

    parser.add_argument(
        "--bold_to_anat_affine",
        default=None,
        type=str,
        help="Manual input of bold-to-anatomical affine",
    )

    parser.add_argument(
        "--bold_to_anat_dof",
        default=6,
        type=bold_to_anat_dof,
        help="Specify degrees-of-freedom for bold-to-anatomical registration (6, 7, 9, 12)\nDefault=6",
    )

    parser.add_argument(
        "--omp_nthreads",
        default=8,
        type=int,
        help="number of threads.",
    )

    """
    Config parameters
    """
    # Processing

    """
    Debug changes
    """
    parser.add_argument(
        "--reg_quick",
        action="store_true",
        help=("[debug] Processes all BOLD runs with only the first 10" " volumes."),
    )

    # Other
    """
    parser.add_argument(
        "--reg_wholebrain_to_anat_dof",
        default=9,
        type=int,
        help=(
            "[registration] Specify DOF for estimating wholebrain epi"
            " to anat registrations. default=9."
        ),
    )

    parser.add_argument(
        "--reg_slab_to_wholebrain_dof",
        default=6,
        type=int,
        help=(
            "[registration] Specify DOF for estimating slab to"
            " wholebrain epi registrations. default=6."
        ),
    )
    """

    return parser
