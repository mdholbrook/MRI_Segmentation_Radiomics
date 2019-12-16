import sys
import os
from glob2 import glob
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.morphology import ball, binary_dilation, binary_closing


def get_foldernames(base_path, no_small=False):

    walker = [x[0] for x in os.walk(base_path)]
    walker.remove(base_path)

    # Get subdirectory names
    subdirs = [os.path.split(i) for i in walker]

    # Select folders with all-numeric names
    paths = [name for (i, name) in enumerate(subdirs)
             if all(char.isdigit() for char in name[1])]

    # Remove folders without 2 weeks of data
    if no_small:
        paths = [name for name in paths if len(name[1]) > 3]

    return [os.path.join(p[0], p[1]) for p in paths]


def get_filenames(path, regexp):

    image_path = os.path.join(path, regexp)

    # Get file names
    names = glob(image_path)

    return names


def gen_savename(files, save_base_path, save_type, ext='.nii'):

    sfiles = []
    ids = []
    for file in files:

        # Get basepath and animal id
        base_path, _ = os.path.split(file[0])
        base_path, id = os.path.split(base_path)

        # save_base_path = os.path.join(base_path, 'Radiomics_16bit')

        sfiles.append(os.path.join(save_base_path, id + save_type + ext + '.gz'))
        ids.append(id)

    return sfiles, ids


def rewrite_nifti(files, sfiles):

    for i in range(len(files)):

        # Load file
        X = nib.load(files[i][0]).get_data().astype(np.float32)

        # Normalize if image
        if len(np.unique(X)) > 3:
            X = (X - X.min()) / (X.max() - X.min())
            X = (2**16 - 1) * X

        X = X.astype(np.uint16)

        # Rewrite file
        nib.save(nib.Nifti1Image(X, np.eye(4)), sfiles[i])


def dilate_masks(mask_files, im_files, dilate_kernel_size, diff=False):
    print('Dilating tumor masks')

    sfiles = []

    for i in range(len(mask_files)):
        print('\t%s' % mask_files[i])

        # Generate updated filename
        path, ext1 = os.path.splitext(mask_files[i])
        path, ext2 = os.path.splitext(path)
        path, filename = os.path.split(path)
        sfile = os.path.join(path, filename + '_dilated' + ext2 + ext1)


        # Load file
        X_mask = nib.load(mask_files[i]).get_data().astype(np.bool)
        X = nib.load(im_files[i][0]).get_data().astype(np.float32)

        # Dilate the mask
        selem = ball(dilate_kernel_size)
        X_dia = binary_dilation(X_mask, selem)

        # Remove air in the mask
        # fig, ax = plt.subplots(1, 3, sharey=True)
        # ax[0].imshow(X[:, :, 35], cmap='gray')
        # ax[0].imshow(np.ma.masked_where(X_mask == 0, X_mask)[:, :, 35], alpha=0.5, cmap='summer')
        # ax[0].set_title('Original mask')
        #
        # ax[1].imshow(X[:, :, 35], cmap='gray')
        # ax[1].imshow(np.ma.masked_where(X_dia == 0, X_dia)[:, :, 35], alpha=0.5, cmap='summer')
        # ax[1].set_title('Dilated mask')
        #
        # X_dia[X < 700] = 0
        # X_dia = binary_closing(X_dia, selem=selem)
        #
        # ax[2].imshow(X[:, :, 35], cmap='gray')
        # ax[2].imshow(np.ma.masked_where(X_dia == 0, X_dia)[:, :, 35], alpha=0.5, cmap='summer')
        # ax[2].set_title('Thresholded, closed mask')
        #
        # plt.savefig(os.path.join(path, filename + '.png'))
        # plt.close(fig)

        if diff:
            X_dia = np.logical_xor(X_dia, X_mask)

        # Save the file
        nib.save(nib.Nifti1Image(X_dia.astype(np.uint16), np.eye(4)), sfile)

        sfiles.append(sfile)

    return sfiles


def write_csv(csv_file, id, image_file, mask_file):

    f = open(csv_file, 'w')
    f.write('ID,Image,Mask\n')

    for i in range(len(id)):
        write_str = id[i] + ',' + image_file[i] + ',' + mask_file[i] + '\n'
        f.write(write_str)


def gen_images_csv(base_path, save_base_path, dilate, ncontrasts=1, regen=True, diff_mask=False):
    """
    Compiles images to be processed into a csv for radiomic processing
    TODO: add multi-contrast processing

    Args:
        base_path (str): path to the image base directories
        save_base_path (str): path in which to save images
        dilate (int): mask dilation radius, no dilation is performed if dilate=0
        ncontrasts (int): the number of image contrasts to use, only 1 or 3 are allowed
        regen (bool): whether or not to re-write image files and masks as uint16

    Returns:
        csv_file (str): path to the csv file containing images to be processed
    """

    # Make sure save path exists
    if not os.path.exists(save_base_path):
        os.mkdir(save_base_path)

    # Dilation setting for masks - UPDATE VALUES
    if dilate == 0:
        dilate = False
        dilate_rad = 0
    else:
        dilate = True
        dilate_rad = 10

    # Get subdirs of the base paths
    subdirs = get_foldernames(base_path, no_small=True)

    if ncontrasts == 1:
        # Get filenames
        image1_regexp = '*1_T2*'
        image2_regexp = '*2_T2*'
        label1_regexp = '*1-label.nii'
        label2_regexp = '*2-label.nii'

        image1 = [get_filenames(sub, image1_regexp) for sub in subdirs]
        image2 = [get_filenames(sub, image2_regexp) for sub in subdirs]
        label1 = [get_filenames(sub, label1_regexp) for sub in subdirs]
        label2 = [get_filenames(sub, label2_regexp) for sub in subdirs]

        # Get save names and ids
        simage1, id = gen_savename(image1, save_base_path, save_type='_T2_1')
        simage2, _ = gen_savename(image2, save_base_path, save_type='_T2_2')
        slabel1, _ = gen_savename(label1, save_base_path, save_type='_mask_1')
        slabel2, _ = gen_savename(label2, save_base_path, save_type='_mask_2')

        # Rewrite to 16 bit
        if regen:
            print('Re-writing images as 16-bit')

            rewrite_nifti(image1, simage1)
            rewrite_nifti(image2, simage2)
            rewrite_nifti(label1, slabel1)
            rewrite_nifti(label2, slabel2)

        # Dilate masks
        if dilate:
            slabel1 = dilate_masks(slabel1, image1, dilate_kernel_size=dilate_rad, diff=diff_mask)
            slabel2 = dilate_masks(slabel2, image2, dilate_kernel_size=dilate_rad, diff=diff_mask)

        # Write CSV file
        csv_file = os.path.join(save_base_path, 'radiomics_files.csv')
        write_csv(csv_file, 2 * id, simage1 + simage2, slabel1 + slabel2)

    elif ncontrasts == 3:

        # Get filenames
        image1T1_regexp = '*1_T1.nii'
        image1T1c_regexp = '*1_T1*C*'
        image1T2_regexp = '*1_T2*'
        image2T1_regexp = '*2_T1.nii'
        image2T1c_regexp = '*2_T1*C*'
        image2T2_regexp = '*2_T2*'
        label1_regexp = '*1-label.nii'
        label2_regexp = '*2-label.nii'

        image1T1 = [get_filenames(sub, image1T1_regexp) for sub in subdirs]
        image1T1c = [get_filenames(sub, image1T1c_regexp) for sub in subdirs]
        image1T2 = [get_filenames(sub, image1T2_regexp) for sub in subdirs]
        image2T1 = [get_filenames(sub, image2T1_regexp) for sub in subdirs]
        image2T1c = [get_filenames(sub, image2T1c_regexp) for sub in subdirs]
        image2T2 = [get_filenames(sub, image2T2_regexp) for sub in subdirs]
        label1 = [get_filenames(sub, label1_regexp) for sub in subdirs]
        label2 = [get_filenames(sub, label2_regexp) for sub in subdirs]

        # Get save names and ids
        simage1T1, id = gen_savename(image1T1, save_base_path, save_type='_T1_1')
        simage1T1c, _ = gen_savename(image1T1c, save_base_path, save_type='_T1c_1')
        simage1T2, _ = gen_savename(image1T2, save_base_path, save_type='_T2_1')
        simage2T1, _ = gen_savename(image2T1, save_base_path, save_type='_T1_2')
        simage2T1c, _ = gen_savename(image2T1c, save_base_path, save_type='_T1c_2')
        simage2T2, _ = gen_savename(image2T2, save_base_path, save_type='_T2_2')
        slabel1, _ = gen_savename(label1, save_base_path, save_type='_mask_1')
        slabel2, _ = gen_savename(label2, save_base_path, save_type='_mask_2')

        # Rewrite to 16 bit
        if regen:
            print('Re-writing images as 16-bit')
            rewrite_nifti(image1T1, simage1T1)
            rewrite_nifti(image1T1c, simage1T1c)
            rewrite_nifti(image1T2, simage1T2)
            rewrite_nifti(image2T1, simage2T1)
            rewrite_nifti(image2T1c, simage2T1c)
            rewrite_nifti(image2T2, simage2T2)
            rewrite_nifti(label1, slabel1)
            rewrite_nifti(label2, slabel2)

        # Dilate masks
        if dilate and regen:
            slabel1 = dilate_masks(slabel1, image1T2, dilate_kernel_size=dilate_rad, diff=diff_mask)
            slabel2 = dilate_masks(slabel2, image2T2, dilate_kernel_size=dilate_rad, diff=diff_mask)

        # Write CSV file
        csv_file = os.path.join(save_base_path, 'radiomics_files_multi.csv')
        write_csv(csv_file,
                  6 * id, simage1T1 + simage1T1c + simage1T2 + simage2T1 + simage2T1c + simage2T2,
                  3*slabel1 + 3*slabel2)

    else:

        raise ValueError('INVALID NUMBER OF CONTRASTS: Enter 1 or 3 contrasts')

    return csv_file


def gen_images_csv_t1c(base_path, save_base_path, dilate, ncontrasts=1, regen=True, diff_mask=False):
    """
    Compiles images to be processed into a csv for radiomic processing
    TODO: add multi-contrast processing

    Args:
        base_path (str): path to the image base directories
        save_base_path (str): path in which to save images
        dilate (int): mask dilation radius, no dilation is performed if dilate=0
        ncontrasts (int): the number of image contrasts to use, only 1 or 3 are allowed
        regen (bool): whether or not to re-write image files and masks as uint16

    Returns:
        csv_file (str): path to the csv file containing images to be processed
    """

    # Make sure save path exists
    if not os.path.exists(save_base_path):
        os.mkdir(save_base_path)

    # Dilation setting for masks - UPDATE VALUES
    if dilate == 0:
        dilate = False
        dilate_rad = 0
    else:
        dilate = True
        dilate_rad = 10

    # Get subdirs of the base paths
    subdirs = get_foldernames(base_path, no_small=True)

    # Get filenames
    image1_regexp = '*1_T1*C*'
    image2_regexp = '*2_T1*C*'
    label1_regexp = '*1-label.nii'
    label2_regexp = '*2-label.nii'

    image1 = [get_filenames(sub, image1_regexp) for sub in subdirs]
    image2 = [get_filenames(sub, image2_regexp) for sub in subdirs]
    label1 = [get_filenames(sub, label1_regexp) for sub in subdirs]
    label2 = [get_filenames(sub, label2_regexp) for sub in subdirs]

    # Get save names and ids
    simage1, id = gen_savename(image1, save_base_path, save_type='_T1C_1')
    simage2, _ = gen_savename(image2, save_base_path, save_type='_T1C_2')
    slabel1, _ = gen_savename(label1, save_base_path, save_type='_mask_1')
    slabel2, _ = gen_savename(label2, save_base_path, save_type='_mask_2')

    # Rewrite to 16 bit
    if regen:
        print('Re-writing images as 16-bit')

        rewrite_nifti(image1, simage1)
        rewrite_nifti(image2, simage2)
        rewrite_nifti(label1, slabel1)
        rewrite_nifti(label2, slabel2)

    # Dilate masks
    if dilate:
        slabel1 = dilate_masks(slabel1, image1, dilate_kernel_size=dilate_rad, diff=diff_mask)
        slabel2 = dilate_masks(slabel2, image2, dilate_kernel_size=dilate_rad, diff=diff_mask)

    # Write CSV file
    csv_file = os.path.join(save_base_path, 'radiomics_files.csv')
    write_csv(csv_file, 2 * id, simage1 + simage2, slabel1 + slabel2)



    return csv_file


if __name__ == "__main__":

    # Dilation setting for masks - UPDATE VALUES
    dilate = True
    dilate_rad = 10

    # Paths to base data (float format)
    if sys.platform == 'linux':
        base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'
        save_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/Radiomics_16bit'

    else:
        base_path = 'E:\\MR Data\\MR_Images_Sarcoma'
        save_base_path = 'E:\\MR Data\\MR_Images_Sarcoma\\Radiomics_16bit'

    # Get subdirs of the base paths
    subdirs = get_foldernames(base_path, no_small=True)

    # Get filenames
    image1_regexp = '*1_T2*'
    image2_regexp = '*2_T2*'
    label1_regexp = '*1-label.nii'
    label2_regexp = '*2-label.nii'

    image1 = [get_filenames(sub, image1_regexp) for sub in subdirs]
    image2 = [get_filenames(sub, image2_regexp) for sub in subdirs]
    label1 = [get_filenames(sub, label1_regexp) for sub in subdirs]
    label2 = [get_filenames(sub, label2_regexp) for sub in subdirs]

    # Get save names and ids
    simage1, id = gen_savename(image1, save_base_path, save_type='_T2_1')
    simage2, _ = gen_savename(image2, save_base_path, save_type='_T2_2')
    slabel1, _ = gen_savename(label1, save_base_path, save_type='_mask_1')
    slabel2, _ = gen_savename(label2, save_base_path, save_type='_mask_2')

    # Rewrite to 16 bit
    rewrite_nifti(image1, simage1)
    rewrite_nifti(image2, simage2)
    rewrite_nifti(label1, slabel1)
    rewrite_nifti(label2, slabel2)

    # Dilate masks
    if dilate:
        slabel1 = dilate_masks(slabel1, image1, dilate_kernel_size=dilate_rad)
        slabel2 = dilate_masks(slabel2, image2, dilate_kernel_size=dilate_rad)

    # Write CSV file
    csv_file = os.path.join(save_base_path, 'radiomics_files.csv')
    write_csv(csv_file, 2*id, simage1 + simage2, slabel1 + slabel2)
    # write_csv(csv_file, id[:2], simage1[:2], slabel1[:2])
