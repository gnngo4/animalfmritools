import argparse


def setup_parser() -> argparse.ArgumentParser:
    """
    Set-up Python's ArgumentParser for oscprep
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
        help="species ID - affects which template is chosen.\nOnly mouse, marmoset are supported.",
    )

    parser.add_argument(
        "--repetition_time",
        default=None,
        type=float,
        help="Manual input repetition time (TR).\nMust specify if json file is missing",
    )

    parser.add_argument(
        "--scratch_dir",
        default="/tmp",
        type=str,
        help="workflow output directory.",
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
