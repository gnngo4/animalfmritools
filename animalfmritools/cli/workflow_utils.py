from pathlib import Path
from typing import Dict, List, Tuple

from base_utils import WorkflowManager
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from animalfmritools.utils.data_grabber import PE_DIR_FLIP


class BufferNodes:
    def __init__(
        self,
        anat: pe.Node,
        template: pe.Node,
        surfaces: pe.Node,
        bold: Dict[str, pe.Node],
        bold_inputs: Dict[str, List[str]],
        fmap: Dict[str, pe.Node],
        fmap_inputs: Dict[str, List[str]],
        bold_session: Dict[str, pe.Node],
        bold_session_template: pe.Node,
        bold_session_template_reg: pe.Node,
    ) -> None:
        self.anat = anat
        self.template = template
        self.surfaces = surfaces
        self.bold = bold
        self.bold_inputs = bold_inputs
        self.fmap = fmap
        self.fmap_inputs = fmap_inputs
        self.bold_session = bold_session
        self.bold_session_template = bold_session_template
        self.bold_session_template_reg = bold_session_template_reg


def get_run_level_buffer_nodes(
    run_dict: Dict[str, List[Path]], image_type: str
) -> Tuple[Dict[str, pe.Node], Dict[str, List[str]]]:
    buffer, buffer_inputs = {}, {}
    for run_type, runs in run_dict.items():
        n_runs = len(runs)
        if n_runs == 0:
            continue
        assert run_type in PE_DIR_FLIP.keys(), f"{run_type} not found."

        buffer_fieldnames = [f"{image_type}_run_{str(ix).zfill(4)}" for ix in range(n_runs)]
        buffer_inputs[run_type] = buffer_fieldnames
        buffer[run_type] = pe.Node(
            niu.IdentityInterface(buffer_fieldnames),
            name=f"{image_type}_{run_type}_buffer",
        )

    return (buffer, buffer_inputs)


def setup_buffer_nodes(wf_manager: WorkflowManager) -> BufferNodes:
    # Native anatomical buffer
    anat_buffer = pe.Node(niu.IdentityInterface(["t2w"]), name="anat_buffer")
    anat_buffer.inputs.t2w = wf_manager.anat

    # Template anatomical buffer
    template_buffer = pe.Node(niu.IdentityInterface(["template", "gm", "wm", "csf"]), name="template_buffer")
    template_buffer.inputs.template = wf_manager.template["Base"]
    template_buffer.inputs.gm = wf_manager.template["Grey"]
    template_buffer.inputs.wm = wf_manager.template["White"]
    template_buffer.inputs.csf = wf_manager.template["CSF"]

    # Surface buffer
    surface_inputs = [
        "lh_midthickness",
        "rh_midthickness",
        "lh_white",
        "rh_white",
        "lh_pial",
        "rh_pial",
        "lh_cortex",
        "rh_cortex",
    ]
    surface_buffer = pe.Node(niu.IdentityInterface(surface_inputs), name="surface_buffer")
    surface_buffer.inputs.lh_midthickness = wf_manager.surfaces["lh_midthickness"]
    surface_buffer.inputs.rh_midthickness = wf_manager.surfaces["rh_midthickness"]
    surface_buffer.inputs.lh_white = wf_manager.surfaces["lh_white"]
    surface_buffer.inputs.rh_white = wf_manager.surfaces["rh_white"]
    surface_buffer.inputs.lh_pial = wf_manager.surfaces["lh_pial"]
    surface_buffer.inputs.rh_pial = wf_manager.surfaces["rh_pial"]
    surface_buffer.inputs.lh_cortex = wf_manager.surfaces["lh_cortex"]
    surface_buffer.inputs.rh_cortex = wf_manager.surfaces["rh_cortex"]

    # bold - run-level - buffers
    bold_buffer, bold_buffer_inputs = get_run_level_buffer_nodes(wf_manager.bold_runs, "bold")

    # fmap - run-level - buffers
    fmap_buffer, fmap_buffer_inputs = get_run_level_buffer_nodes(wf_manager.fmap_runs, "fmap")

    # bold - [session|PE]-level - buffers
    bold_session_buffer = {}
    for run_type, _runs in bold_buffer_inputs.items():
        bold_session_buffer[run_type] = pe.Node(
            niu.IdentityInterface(["sdc_warp", "sdc_affine", "sdc_bold", "distorted_bold"]),
            name=f"bold_session_{run_type}_buffer",
        )

    # bold - session-level - buffer
    bold_session_template_buffer = pe.Node(
        niu.IdentityInterface(["bold_session_template"]),
        name="bold_session_template_buffer",
    )

    # reg - [session|PE]-to-session template - affine registrations
    bold_session_template_reg_buffer = pe.Node(
        niu.IdentityInterface(
            [f"bold_session_{run_type}_to_bold_session_template_reg" for run_type in bold_session_buffer.keys()]
        ),
        name="bold_session_template_reg_buffer",
    )

    # Build dictionary
    buffer_nodes = BufferNodes(
        anat=anat_buffer,
        template=template_buffer,
        surfaces=surface_buffer,
        bold=bold_buffer,
        bold_inputs=bold_buffer_inputs,
        fmap=fmap_buffer,
        fmap_inputs=fmap_buffer_inputs,
        bold_session=bold_session_buffer,
        bold_session_template=bold_session_template_buffer,
        bold_session_template_reg=bold_session_template_reg_buffer,
    )

    return buffer_nodes


# Workflow functions
def load_json_as_dict(json_path):
    import json

    assert json_path.exists(), f"{json_path} not found."
    with open(json_path) as json_file:
        metadata = json.load(json_file)

    return metadata


def pick_rel(rms_files):
    return rms_files[-1]


def jsonify(_dict):
    import json

    filename = "/tmp/confounds_metadata.json"

    with open(filename, "w") as f:
        json.dump(_dict, f)

    return filename


def listify_three_inputs(input_1, input_2, input_3):
    return [input_1, input_2, input_3]


def get_split_volume(out_files, vol_id):
    return out_files[vol_id]
