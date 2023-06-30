from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_bold_ref_wf(
	split_vol_idx: int = 0, 
	name: str = "bold_reference_wf",
):
	
	from nipype.interfaces.fsl.utils import Split
	
	workflow = Workflow(name=name)
	
	inputnode = pe.Node(
		niu.IdentityInterface(["bold"]),
		name = "inputnode",
	)

	outputnode = pe.Node(
		niu.IdentityInterface(["boldref"]),
		name = "outputnode",
	)

	split_bold = pe.Node(
		Split(dimension="t", out_base_name="split_bold_"),
		name = "split_bold"
	)

	# fmt: off
	workflow.connect([
		(inputnode, split_bold, [("bold", "in_file")]),
		(split_bold, outputnode, [(("out_files", _get_split_volume, split_vol_idx), "boldref")])
	])
	# fmt: on

	return workflow

def _get_split_volume(out_files, vol_idx):
	return out_files[vol_idx]