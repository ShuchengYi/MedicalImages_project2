import SimpleITK as sitk

def est_lin_transf(im_ref, im_mov, mask_ref=None, mask_mov=None, verbose=False):
    """
    Estimate affine transform to align im_mov to im_ref.
    Optionally uses masks to focus registration on bone.
    Returns a SimpleITK Transform.
    """
    init_transform = sitk.CenteredTransformInitializer(
        im_ref,
        im_mov,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(init_transform, inPlace=False)

    # Metric
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if mask_ref is not None:
        reg.SetMetricFixedMask(mask_ref)
    if mask_mov is not None:
        reg.SetMetricMovingMask(mask_mov)

    # Optimizer
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Interpolator
    reg.SetInterpolator(sitk.sitkLinear)

    # --- Execute ---
    final_transform = reg.Execute(sitk.Cast(im_ref, sitk.sitkFloat32),
                                  sitk.Cast(im_mov, sitk.sitkFloat32))
    if verbose:
        print(final_transform)
        print("--------")
        print("Optimizer stop condition: {0}".format(reg.GetOptimizerStopConditionDescription()))
        print("Number of iterations: {0}".format(reg.GetOptimizerIteration()))
        print("Final metric value:", reg.GetMetricValue())
        print("--------")

    return final_transform

def apply_lin_transf(im_mov, lin_xfm, im_ref, is_mask=False):
    """
    Apply linear transform to im_mov and resample into im_ref space.
    """
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    return sitk.Resample(
        im_mov,
        im_ref,
        lin_xfm,
        interpolator,
        0,
        im_mov.GetPixelID()
    )
