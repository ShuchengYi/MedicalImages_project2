from ipywidgets import interact, IntSlider, fixed
import matplotlib.pyplot as plt
import numpy as np

def show_coronal_slice(arr, slc, mask=None):
    plt.figure(figsize=(5,5))
    plt.imshow(np.flip(arr[:, slc, :],0), cmap='gray')
    if mask is not None:
        plt.imshow(
            np.flip(np.ma.masked_where(mask[:, slc, :] == 0, mask[:, slc, :]),0),
            cmap='hsv',
            vmin=0,
            vmax=5,
            alpha=0.3
        )
    plt.axis('off')
    plt.title(f'Coronal slice {slc}')
    plt.show()

def show_axial_slice(arr, slc, mask=None):
    plt.figure(figsize=(5,5))
    plt.imshow(np.flip(arr[slc, :, :],0), cmap='gray')
    if mask is not None:
        plt.imshow(
            np.flip(np.ma.masked_where(mask[slc, :, :] == 0, mask[slc, :, :]),0),
            cmap='hsv',
            vmin=0,
            vmax=5,
            alpha=0.3
        )
    plt.axis('off')
    plt.title(f'Axial slice {slc}')
    plt.show()

def show_sagital_slice(arr, slc, mask=None):
    plt.figure(figsize=(5,5))
    plt.imshow(np.flip(arr[:, :, slc],0), cmap='gray')
    if mask is not None:
        plt.imshow(
            np.flip(np.ma.masked_where(mask[:, :, slc] == 0, mask[:, :, slc]),0),
            cmap='hsv',
            vmin=0,
            vmax=5,
            alpha=0.3
        )
    plt.axis('off')
    plt.title(f'Sagital slice {slc}')
    plt.show()

def show_interactive(arr, plane: str, mask=None):
    """
    Create interactive visualization with slider in on of the anatomical planes
    :param arr: Array containing the complete image to be displayed
    :param plane: Anatomical plane. Can be 'axial', 'coronal' or 'sagital'
    :param mask: If an array containing a segmentation exists, include it here.
    :return: Interactive slider plot.
    """
    match plane:
        case 'axial':
            fn = show_axial_slice
            ax = 0
        case 'coronal':
            fn = show_coronal_slice
            ax = 1
        case 'sagital':
            fn = show_sagital_slice
            ax = 2
        case _:
            raise ValueError(f"Plane {plane} not recognized")
    return interact(
        fn,
        slc=IntSlider(
            min=0,
            max=arr.shape[ax] - 1,
            step=1,
            value=arr.shape[ax] // 2),
        arr=fixed(arr),
        mask=fixed(mask)
    )

def show_coronal_overlay(fix_arr, mov_arr, slc):
    plt.figure(figsize=(5,5))
    
    plt.imshow(np.flip(fix_arr[:, slc, :],0), cmap='Reds')
    plt.imshow(np.flip(mov_arr[:, slc, :],0), cmap='Blues', alpha=0.3)
    
    plt.axis('off')
    plt.title(f'Coronal slice {slc}')
    plt.show()

def show_interactive_overlay(fix_arr, mov_arr, plane: str):
    match plane:
        case 'axial':
            fn = show_coronal_overlay
            ax = 0
        case 'coronal':
            fn = show_coronal_overlay
            ax = 1
        case 'sagital':
            fn = show_coronal_overlay
            ax = 2
        case _:
            raise ValueError(f"Plane {plane} not recognized")
    return interact(
        fn,
        slc=IntSlider(
            min=0,
            max=fix_arr.shape[ax] - 1,
            step=1,
            value=fix_arr.shape[ax] // 2
        ),
        fixed_img=fixed(fix_arr),
        moving_img=fixed(mov_arr)
    )
