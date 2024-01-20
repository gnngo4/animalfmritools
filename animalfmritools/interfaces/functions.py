def get_translation_component_from_nifti(nifti_path):
    import nibabel as nib

    img = nib.load(nifti_path)
    return img.affine[:3, 3]
