import SimpleITK as sitk

def mask_from_hu(img, hu_min=206, closing_kernel_size=4):
    """
    Create a simple mask from HU thresholding.
    img: SimpleITK image (CT en HU)
    hu_min: minimum threshold for target structures
    returns: SimpleITK binary mask (UInt8)
    """
    mask = img >= hu_min
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    # mask = sitk.BinaryMorphologicalOpening(mask, [2,2,2])
    mask = sitk.BinaryMorphologicalClosing(mask, [closing_kernel_size,closing_kernel_size,closing_kernel_size])
    mask = sitk.BinaryFillhole(mask)
    return mask