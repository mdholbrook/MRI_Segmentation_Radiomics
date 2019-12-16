import os
import nibabel as nib
import numpy as np
from overlay_ims import overlay_image
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl


def main():
    """
    This function contains the step for processing and saving radiomic map overlays. This is used on
    previously-calculated radiomic voxel maps.
    """

    # Histogram clipping
    top_pct = 0.5 / 100
    bot_pct = 30 / 100

    # Set up image and mask paths
    path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/Analysis/VoxelRadiomics'
    masks = ['Pre_3_gldm_DependenceNonUniformity.nii.gz',
             'Post_3_gldm_DependenceNonUniformity.nii.gz']
    ims = ['Pre_0_Image.nii', 'Post_0_Image.nii']
    slcs = [35, 32]
    cm = 'inferno'
    alpha = 1

    masks = [os.path.join(path, m) for m in masks]
    ims = [os.path.join(path, m) for m in ims]
    over_im1 = [None] * 2
    mask = [None] * 2


    # Load and normalize maps
    for z in range(len(masks)):
        # Load
        mask[z] = nib.load(masks[z]).get_data().astype(np.float32)
        mask[z] = mask[z][:, :, slcs[z] - 1: slcs[z] + 1].mean(axis=2).T

    mask_min = min(mask[0].min(), mask[1].min())
    mask_max = max(mask[0].max(), mask[1].max())

    for z in range(len(masks)):
        mask[z] = (mask[z] - mask_min) / (mask_max - mask_min)

    # Process images
    for z in range(len(ims)):

        # Load data
        im = nib.load(ims[z]).get_data().astype(np.float32)

        # Select the slice to show
        im = im[:, :, slcs[z] - 1: slcs[z] + 1].mean(axis=2).T

        # Scale the image
        im_s = sorted(im.reshape(-1))
        top_thresh = im_s[-round(len(im_s) * top_pct)]
        bot_thresh = im_s[round(len(im_s) * bot_pct)]
        im[im > top_thresh] = top_thresh
        im[im < bot_thresh] = bot_thresh

        # Create overlay
        over_im = overlay_image(im, mask[z], colormap=cm, alpha=alpha)

        over_im = over_im[range(30, over_im.shape[0]-50), :]
        over_im = over_im[:, range(40, over_im.shape[1]-40)]

        # Convert image to PIL
        over_im1[z] = Image.fromarray(over_im)

        # Create save name
        sname = os.path.splitext(ims[z])[0] + '.png'

        # Save image
        over_im1[z].save(sname, dpi=(300.0, 300.0))

    # Save montage
    lim = Image.new('RGB', size=(2*over_im1[0].width, over_im1[0].height))
    x, y = 0, 0
    for im in over_im1:
        lim.paste(im, (x, 0, x + im.width, y + im.height))
        x += im.width

    lim.save(os.path.join(path, 'concat_3.png'), dpi=(300.0, 300.0))

    # Set up image and mask paths
    masks = ['Pre_2_glszm_LargeAreaLowGrayLevelEmphasis.nii.gz',
             'Post_2_glszm_LargeAreaLowGrayLevelEmphasis.nii.gz']
    ims = ['Pre_0_Image.nii', 'Post_0_Image.nii']
    slcs = [35, 32]
    cm = 'inferno'

    masks = [os.path.join(path, m) for m in masks]
    ims = [os.path.join(path, m) for m in ims]
    over_im1 = [None] * 2
    mask = [None] * 2

    # Load and normalize maps
    for z in range(len(masks)):

        # Load
        mask[z] = nib.load(masks[z]).get_data().astype(np.float32)
        mask[z] = mask[z][:, :, slcs[z] - 1: slcs[z] + 1].mean(axis=2).T

    mask_min = min(mask[0].min(), mask[1].min())
    mask_max = max(mask[0].max(), mask[1].max())

    for z in range(len(masks)):
        mask[z] = (mask[z] - mask_min) / (mask_max - mask_min)

    # Process images
    for z in range(len(ims)):

        # Load data
        im = nib.load(ims[z]).get_data().astype(np.float32)

        # Select the slice to show
        im = im[:, :, slcs[z] - 1: slcs[z] + 1].mean(axis=2).T

        # Scale the image
        im_s = sorted(im.reshape(-1))
        top_thresh = im_s[-round(len(im_s) * top_pct)]
        bot_thresh = im_s[round(len(im_s) * bot_pct)]
        im[im > top_thresh] = top_thresh
        im[im < bot_thresh] = bot_thresh

        # Create overlay
        over_im = overlay_image(im, mask[z], colormap=cm, alpha=alpha)

        # Convert image to PIL
        over_im1[z] = Image.fromarray(over_im)

        # Create save name
        sname = os.path.splitext(ims[z])[0] + '.png'

        # Save image
        over_im1[z].save(sname, dpi=(300.0, 300.0))

    # Save montage
    lim = Image.new('RGB', size=(2*over_im1[0].width, over_im1[0].height))
    x, y = 0, 0
    for im in over_im1:
        lim.paste(im, (x, 0, x + im.width, y + im.height))
        x += im.width

    lim.save(os.path.join(path, 'concat_2.png'), dpi=(300.0, 300.0))

    # Save colorbar
    fig, ax = plt.subplots(figsize=(1, 6))
    # fig.subplots_adjust(bottom=0.5)
    fig.subplots_adjust(right=0.5)

    cmap = mpl.cm.get_cmap(cm)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Normalized Value')
    # plt.tight_layout()

    sname = os.path.join(path, 'colorbar_vert.svg')
    fig.savefig(sname, dpi=200)

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    # fig.subplots_adjust(right=0.5)

    cmap = mpl.cm.get_cmap(cm)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Normalized Value')
    # plt.tight_layout()

    sname = os.path.join(path, 'colorbar_hort.svg')
    fig.savefig(sname, dpi=200)


if __name__ == "__main__":

    main()


