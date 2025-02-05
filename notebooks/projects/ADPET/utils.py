from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import scipy

"""
ABA parser
"""
MOUSE_TEMPLATE_DIR = Path("/opt/animalfmritools/notebooks/projects/ADPET/MouseABA")
TEMPLATE_LABEL_NIFTI = MOUSE_TEMPLATE_DIR / "P56_Annotation_downsample2.nii.gz"
EXPECTED_KEYS_MAIN = ['Basic cell groups and regions', 'fiber tracts', 'ventricular systems', 'grooves', 'retina']

EXPECTED_KEYS_GM = [
    "Cerebrum",
    "Brain stem",
    "Cerebellum",
]

EXPECTED_KEYS_WM = [
    "cranial nerves",
    "cerebellum related fiber tracts",
    "supra-callosal cerebral white matter",
    "lateral forebrain bundle system",
    "extrapyramidal fiber systems",
    "medial forebrain bundle system",
]

EXPECTED_KEYS_VS = [
    "lateral ventricle",
    "interventricular foramen",
    "third ventricle",
    "cerebral aqueduct",
    "fourth ventricle",
    "central canal, spinal cord/medulla",
]


def aba_setup():
    roi_dir = MOUSE_TEMPLATE_DIR / "rois"
    if not roi_dir.exists():
        roi_dir.mkdir()

    return roi_dir


def load_json(json_path: Path) -> Dict:
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def assert_keys(check_dict: Dict, list_keys: List[str]) -> None:
    for k in check_dict.keys():
        assert k in list_keys


def organize_substructures(aba_dict: Dict, target_key: str, expected_keys: List[str]) -> Dict:
    sub_dict = organize_main_structures(aba_dict)[target_key]['children']
    organized_dict = {item['name']: item for item in sub_dict}
    assert_keys(organized_dict, expected_keys)
    return organized_dict


def organize_main_structures(aba_dict: Dict) -> Dict:
    org = aba_dict['msg'][0]['children']
    organized_dict = {item['name']: item for item in org}
    assert_keys(organized_dict, EXPECTED_KEYS_MAIN)
    return organized_dict


def organize_gm_structures(aba_dict: Dict) -> Dict:
    return organize_substructures(aba_dict, 'Basic cell groups and regions', EXPECTED_KEYS_GM)


def organize_wm_structures(aba_dict: Dict) -> Dict:
    return organize_substructures(aba_dict, 'fiber tracts', EXPECTED_KEYS_WM)


def organize_vs_structures(aba_dict: Dict) -> Dict:
    return organize_substructures(aba_dict, 'ventricular systems', EXPECTED_KEYS_VS)


def organize_aba_structures(json_file) -> Dict:
    aba_dict = load_json(json_file)
    main = organize_main_structures(aba_dict)
    gm = organize_gm_structures(aba_dict)
    wm = organize_wm_structures(aba_dict)
    vs = organize_vs_structures(aba_dict)

    return {
        "main": main,
        "gm": gm,
        "wm": wm,
        "vs": vs,
    }


def extract_levels(
    children: Dict,
    extract_level: int,
    level: int = 1,
    graph_idxs: Dict = {},
    main_key=None,
    verbose=False,
):
    from copy import deepcopy

    graph_idxs = deepcopy(graph_idxs)

    for child in children:
        k_tuple = (child['name'], child['acronym'])
        v_tuple = (
            child['name'],
            child['acronym'],
            child['graph_order'],
            child['color_hex_triplet'],
            child['st_level'],
            child['ontology_id'],
        )
        if extract_level == level:
            graph_idxs[k_tuple] = [v_tuple]
            main_key = k_tuple
            tracker = 'x'
        elif level > extract_level:
            graph_idxs[main_key].append(v_tuple)
            tracker = '>'
        else:
            tracker = 'o'

        if verbose:
            print(
                f"{tracker} [{str(child['graph_order']).zfill(4)}] {'-'*level} {child['name']} [{child['acronym']}] {level}"
            )

        if 'children' in child:
            graph_idxs = extract_levels(child['children'], extract_level, level + 1, graph_idxs, main_key, verbose)

    return graph_idxs


def check_template_for_idx(template_idx: int, template_nifti: str = TEMPLATE_LABEL_NIFTI) -> bool:
    data = nib.load(template_nifti).get_fdata()
    n_voxels_with_idx = np.where(data == template_idx)[0].shape[0]
    if n_voxels_with_idx > 0:
        return True
    else:
        return False


def get_template_coords_from_idx(template_idx: int, template_nifti: str = TEMPLATE_LABEL_NIFTI) -> Tuple:
    data = nib.load(template_nifti).get_fdata()
    coords = np.where(data == template_idx)

    return coords


def save_template_roi(
    parent_structure_label: str,
    roi_label: str,
    template_coords: Tuple,
    outdir: Path,
    template_nifti: str = TEMPLATE_LABEL_NIFTI,
) -> None:
    template_img = nib.load(template_nifti)
    template_data = template_img.get_fdata()
    roi_data = np.zeros(template_data.shape)
    roi_data[(template_coords[0, :], template_coords[1, :], template_coords[2, :])] = 1
    roi_img = nib.Nifti1Image(
        roi_data,
        header=template_img.header,
        affine=template_img.affine,
    )
    output_path = outdir / f"P56_desc-{parent_structure_label}_roi-{roi_label}.nii.gz"
    if not Path(output_path).exists():
        print(f"Saving to {output_path}.")
        nib.save(roi_img, output_path)
    else:
        print(f"{output_path} already exists.")


def parse_children(
    children,
    level=1,
    parent_structure=None,
    previous_structure_label=None,
    previous_structure_name=None,
    structure_mapping=None,
    roi_hierarchy=None,
    outdir: Path = None,
):
    if structure_mapping is None:
        structure_mapping = dict()

    if roi_hierarchy is None:
        roi_hierarchy = dict()

    if len(children) > 0:
        # print('ROI includes: ')
        exist_first = False
        create_roi = False
        for c in children:
            # Fill structure mapping
            structure_mapping[c['acronym']] = c['name']

            # Fill roi hierarchy mapping
            if previous_structure_label is not None:
                if previous_structure_label not in roi_hierarchy.keys():
                    roi_hierarchy[previous_structure_label] = [c['acronym']]
                else:
                    roi_hierarchy[previous_structure_label].append(c['acronym'])

            # Check if nifti label exists in the atlas
            nifti_label = c['graph_order']
            label_exists = check_template_for_idx(nifti_label)
            # Print info
            if label_exists:
                create_roi = True
                # print(f"{'-'*level} [{c['acronym']}] {c['name']} {c['color_hex_triplet']} {label_exists} || PRIOR: [{previous_structure_label}] {previous_structure_name}")
                coords = get_template_coords_from_idx(nifti_label)
                if not exist_first:
                    joined_coords = np.vstack(coords)
                    exist_first = True
                else:
                    joined_coords = np.concatenate((coords, joined_coords), axis=1)
            else:
                pass
            structure_mapping, roi_hierarchy = parse_children(
                c['children'],
                level=level + 1,
                parent_structure=parent_structure,
                previous_structure_label=c['acronym'],
                previous_structure_name=c['name'],
                structure_mapping=structure_mapping,
                roi_hierarchy=roi_hierarchy,
                outdir=outdir,
            )

        if create_roi:
            save_template_roi(parent_structure, previous_structure_label, joined_coords, outdir)

    return structure_mapping, roi_hierarchy


"""
ABA ROI sorter
"""


def get_roi_path(roi_dir, roi_acronym, parent_acronym='CH'):
    roi_path = roi_dir / f"P56_desc-{parent_acronym}_roi-{roi_acronym}.nii.gz"
    if roi_path.exists():
        return roi_path


def search_for_roi_paths(roi_dir, all_label_hierarchies, main_k, sub_k, parent_acronym, roi_paths=None):
    all_keys = [i for i in all_label_hierarchies[main_k].keys()]

    if roi_paths is None:
        roi_paths = []

    for s in all_label_hierarchies[main_k][sub_k]:
        roi_path = get_roi_path(roi_dir, s, parent_acronym)
        if s not in all_keys:
            continue

        if roi_path is not None:
            roi_paths.append(roi_path)
            roi_paths = search_for_roi_paths(
                roi_dir, all_label_hierarchies, main_k, s, parent_acronym, roi_paths=roi_paths
            )
        else:
            roi_paths = search_for_roi_paths(
                roi_dir, all_label_hierarchies, main_k, s, parent_acronym, roi_paths=roi_paths
            )

    return roi_paths


"""
Atlas functions
"""


def extract_roi_name(nifti_path: Path) -> str:
    nifti_path = str(nifti_path)
    assert '_roi-' in nifti_path

    return str(nifti_path).split('_roi-')[1].split('.nii.gz')[0]


def get_empty_template(nifti_path: str) -> np.ndarray:
    return np.zeros(nib.load(nifti_path).shape)


def create_roi_array(roi_index, roi_path, hemi_path=None, hemi_idx=None) -> np.ndarray:
    # Load data
    # import pdb; pdb.set_trace()

    roi_data = (nib.load(roi_path).get_fdata() > 0).astype(int)

    if hemi_path is not None and hemi_idx is not None:
        hemi_data = nib.load(hemi_path).get_fdata()
        # Remove opposite hemisphere depending on label
        hemi_data[np.where(hemi_data != hemi_idx)] = 0
        hemi_data[np.where(hemi_data == hemi_idx)] = 1
        # Filter roi data with hemisphere
        roi_data = roi_data * hemi_data

    # Label roi data
    roi_data *= roi_index

    return roi_data


def create_atlas(
    roi_nifti_list: List, roi_parent_list: List, hemi_nifti: str, out_dir: str = "/tmp/mouse_atlas.nii.gz"
):
    # `hemi_nifti` annotates RH with idx == 2 and LH with idx == 1
    hemi_mapping = {
        'RH': 2,
        'LH': 1,
    }

    # Instantiate empty atlas array (to be filled in)
    merged_atlas = get_empty_template(roi_nifti_list[0])
    atlas_annotations = []  # Set up empty annotation list

    # for ix, (roi_nifti, roi_parent_label) in enumerate(zip(roi_nifti_list, roi_parent_list)):
    import itertools

    for ix, ((roi_nifti, roi_parent_label), hemi_label) in enumerate(
        itertools.product(zip(roi_nifti_list, roi_parent_list), hemi_mapping.keys())
    ):  # Considers hemi
        print(ix, roi_nifti, roi_parent_label)

        ix = ix + 1
        # hemi_idx = hemi_mapping[hemi_label]

        # All niftis must be the same shape as the empty atlas array
        assert nib.load(roi_nifti).shape == merged_atlas.shape

        # Add reannotated ROI to a single array
        merged_atlas += create_roi_array(
            ix, roi_nifti, hemi_path=hemi_nifti, hemi_idx=hemi_mapping[hemi_label]
        )  # hemi_* set to None turns off hemispheric parsing
        # Add label to atlas annotations
        roi_label = extract_roi_name(roi_nifti)
        atlas_annotations.append((ix, roi_parent_label, f"{roi_label}_{hemi_label}"))  # considers hemi
        # atlas_annotations.append((ix, roi_parent_label, f"{roi_label}"))

    # Round
    merged_atlas = np.round(merged_atlas)

    # Save
    template_img = nib.load(roi_nifti_list[0])
    merged_atlas_img = nib.Nifti1Image(merged_atlas, affine=template_img.affine, header=template_img.header)
    nib.save(merged_atlas_img, out_dir)

    return atlas_annotations, out_dir


def main_create_atlas(json_file, roi_dir, mouse_template_dir=MOUSE_TEMPLATE_DIR):
    import json

    def dump_dict_to_json(json_path, _dict):
        with open(json_path, "w") as f:
            json.dump(_dict, f, indent=4)  # Write to file with indentation

    _, gm, wm, vs = organize_aba_structures(json_file).values()
    # Generate ROIs and store to `roi_dir`
    all_label_mappings = {}
    all_label_hierarchies = {}
    for structure in [gm, wm, vs]:
        for structure_ix, (k, v) in enumerate(structure.items()):
            print(f"[{structure_ix + 1}/{len(structure)}] {v['name']} {v['acronym']}")
            parent_label = f"{v['name']} {v['acronym']}"
            label_mapping, label_hierarchy = parse_children(
                v['children'], level=1, parent_structure=v['acronym'], outdir=roi_dir
            )
            all_label_mappings[parent_label] = label_mapping
            all_label_hierarchies[parent_label] = label_hierarchy

    dump_dict_to_json(roi_dir / "label_mappings.json", all_label_mappings)
    dump_dict_to_json(roi_dir / "label_hierarchies.json", all_label_hierarchies)

    # Remove this ROI - it is small and does not include DG or CA
    # MM, CN, VNC was not allocated any voxels after resampling atlas to EPI space
    #!rm {roi_dir}/*roi-HIP* {roi_dir}/*roi-HY* {roi_dir}/*roi-PB* {roi_dir}/*roi-AMB* {roi_dir}/*roi-MY-sat*

    # Get list of ROI paths
    # Cerebrum
    k = "Cerebrum CH"
    isocortex = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "Isocortex", "CH")
    olf = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "OLF", "CH")
    hpf = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "HPF", "CH")
    # Brain stem
    k = 'Brain stem BS'
    interbrain = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "IB", "BS")
    midbrain = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "MB", "BS")
    hindbrain = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "HB", "BS")
    # Cerebellum
    k = 'Cerebellum CB'
    cerebellar_cortex = search_for_roi_paths(roi_dir, all_label_hierarchies, k, "CBX", "CB")

    # Create atlas
    hemi_nifti = mouse_template_dir / "roi-hemispheres_ABAv3.nii.gz"  # RH == 2 & LH == 1
    atlas_annots, atlas_nifti = create_atlas(
        isocortex + hpf + interbrain + midbrain + hindbrain,
        ["Isocortex"] * len(isocortex)
        + ["HPF"] * len(hpf)
        + ["Interbrain"] * len(interbrain)
        + ["Midbrain"] * len(midbrain)
        + ["Hindbrain"] * len(hindbrain),
        hemi_nifti,
    )

    return atlas_annots, atlas_nifti


def check_labels_in_atlas(atlas_annots, atlas_path):
    atlas_data = np.round(nib.load(atlas_path).get_fdata())
    unique_labels = np.unique(atlas_data)
    for roi_idx, _, roi_label in atlas_annots:
        if roi_idx not in unique_labels:
            print(f"[WARNING] {roi_label} [{roi_idx}] not in atlas.")


def calculate_mean_tsnr(timeseries: np.ndarray) -> float:
    _values = timeseries.mean(axis=1) / timeseries.std(axis=1)
    # Create a boolean mask to identify NaN values
    nan_mask = np.isnan(_values)
    # Filter out NaN values
    _values = _values[~nan_mask]
    mean_value = _values.mean()

    return mean_value


def alff_and_falff(x, sampling_rate, low_freq_range=(0.01, 0.1)):
    from scipy.signal import welch

    fs, power = welch(x, fs=sampling_rate)
    # get indices corresponding to `low_freq_range`
    low_freq_idx = np.where((fs >= low_freq_range[0]) & (fs <= low_freq_range[1]))
    # calculate ALFF: square root of the average power in the low-frequency range
    alff = np.sqrt(np.mean(power[low_freq_idx]))
    # calculate fALFF: ALFF / total power across all frequencies
    total_power = np.sum(power)
    falff = alff / total_power

    return alff, falff


def lag_1_ta(x):
    return np.corrcoef(x[0:-1], x[1:])[0, 1]


def correlation_matrix_to_dict(C, labels, fisher_Z=True) -> Dict:
    if fisher_Z:
        C = np.arctanh(C)

    n_rois = C.shape[0]
    counter = 0
    conn_dict = {}
    for i in range(n_rois):
        for j in range(n_rois):
            if i > j:
                connection_pair = f"{labels[i]}-{labels[j]}"
                conn_dict[connection_pair] = C[i, j]
                counter += 1

    return conn_dict


def spatial_autocorrelation(cm, dist, discretization=1):
    """Calculate the SA-λ and SA-∞ measures of spatial autocorrelation, defined in [Shinn et al (2023)](https://www.nature.com/articles/s41593-023-01299-3)

    Args:
      cm (NxN numpy array): NxN correlation matrix of timeseries, where N is the number of
          timeseries
      dist (NxN numpy array): the NxN distance matrix, representing the spatial distance
          between location of each of the timeseries.  This should usually be the
          output of the `distance_matrix_euclidean` function.
      discretization (int): The size of the bins to use when computing the SA parameters.
          The size of the discretization should ensure that there are a sufficient number of
          observations in each bin, but also enough total bins to make a meaningful estimation.
          Try increasing it or decreasing it according to the scale of your data.  Data that has values
          up to around 100 should be fine with the default.  Decrease or increase as necessary
          to get an appropriate estimation.

    Returns:
      tuple of floats: tuple of (SA-λ, SA-∞)
    """
    tril_indices = np.tril_indices(dist.shape[0], -1)
    cm_flat = cm[tril_indices]
    dist_flat = dist[tril_indices]
    df = pd.DataFrame(np.asarray([dist_flat, cm_flat]).T, columns=["dist", "corr"])
    df['dist_bin'] = np.round(df['dist'] / discretization) * discretization
    df_binned = df.groupby('dist_bin').mean().reset_index().sort_values('dist_bin')
    binned_dist_flat = df_binned['dist_bin']
    binned_cm_flat = df_binned['corr']
    binned_cm_flat[0] = 1  # Distance of zero should give correlation of 1.
    spatialfunc = lambda v: np.exp(-binned_dist_flat / v[0]) * (1 - v[1]) + v[1]
    with np.errstate(all='warn'):
        res = scipy.optimize.minimize(
            lambda v: np.sum((binned_cm_flat - spatialfunc(v)) ** 2), [10, 0.3], bounds=[(0.1, 100), (-1, 1)]
        )
    return (res.x[0], res.x[1])


def calculate_all_metrics(bold_nifti, atlas_nifti, atlas_annots, TR, distance_matrix, sa_discretization):
    # Load niftis
    bold_img = nib.load(bold_nifti)
    atlas_img = nib.load(atlas_nifti)

    # Check whether images have same coordinate system
    assert np.allclose(bold_img.affine, atlas_img.affine)

    bold_data = bold_img.get_fdata()
    atlas_data = np.round(atlas_img.get_fdata())

    # Extract average timeseries from all labels
    # Includes, ta_lag_1, alff, falff, tsnr
    label_timeseries, label_tsnr, labels = [], [], []
    label_alff, label_falff, label_ta_lag_1 = [], [], []
    for atlas_idx, _, atlas_labels in atlas_annots:
        atlas_coords = np.where(atlas_data == atlas_idx)
        label_all_timeseries = bold_data[atlas_coords]
        label_avg_timeseries = np.mean(label_all_timeseries, axis=0)
        alff, falff = alff_and_falff(label_avg_timeseries, TR)
        ta_lag_1 = lag_1_ta(label_avg_timeseries)
        label_alff.append(alff)  # append alff
        label_falff.append(falff)  # append falff
        label_ta_lag_1.append(ta_lag_1)  # append temporal autocorrelation
        label_avg_timeseries = (
            label_avg_timeseries - label_avg_timeseries.mean()
        ) / label_avg_timeseries.std()  # z-score normalization
        label_timeseries.append(
            label_avg_timeseries
        )  # store all z-scored timeseries -> Compute correlation matrix, outside of loop
        labels.append(atlas_labels)  # store labels
        label_tsnr.append(calculate_mean_tsnr(label_all_timeseries))  # append tsnr
    # Compute RSFC matrix
    C = np.corrcoef(label_timeseries)
    # Calculate spatial autocorrelation - 2 ways as per Shinn et al.
    global_sa_lambda, global_sa_inf = spatial_autocorrelation(C, distance_matrix, discretization=sa_discretization)
    # Global temporal autocorrelation
    global_ta_lag_1 = np.array(label_ta_lag_1).mean()
    # Convert correlation matrix to dict
    C = correlation_matrix_to_dict(C, labels, fisher_Z=False)

    return {
        "labels": labels,
        "tsnr": label_tsnr,
        "alff": label_alff,
        "falff": label_falff,
        "ta_lag_1": label_ta_lag_1,
        "global_ta_lag_1": global_ta_lag_1,
        "global_sa_lambda": global_sa_lambda,
        "global_sa_inf": global_sa_inf,
        "rsfc": C,
    }


def euclidean_distance_from_atlas(atlas_nifti):
    from scipy.spatial import distance

    # Load atlas
    atlas_img = nib.load(atlas_nifti)
    x_size, y_size, z_size = atlas_img.header.get_zooms()  # mm
    atlas_data = np.round(atlas_img.get_fdata())
    labels = np.unique(atlas_data)
    # Calculate mean location of ROI
    mean_locations = []
    for i in range(1, len(labels), 1):
        mean_location = np.mean(np.where(atlas_data == i), axis=1)
        mean_locations.append(mean_location)
    # distance matrix is in voxel unit
    distance_matrix = distance.cdist(mean_locations, mean_locations, 'euclidean')
    # Rescale by atlas dim
    assert x_size == y_size == z_size, f"{x_size} != {y_size} != {z_size}"
    distance_matrix *= x_size

    return distance_matrix
