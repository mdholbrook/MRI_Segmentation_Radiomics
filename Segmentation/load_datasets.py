import os
import pandas as pd
from glob2 import glob
from tqdm import tqdm
from Segmentation.model_keras import *
from Radiomics.Nifti_float_to_16bit import get_foldernames, get_filenames
from adaptive_hist import adaptive_hist_eq_3D


def load_filenames(base_path):

    # Get subdirs
    subdirs = get_foldernames(base_path)

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

    # Concatenate lists
    imageT1 = image1T1 + image2T1
    imageT1c = image1T1c + image2T1c
    imageT2 = image1T2 + image2T2
    label = label1 + label2

    # Flatten the list of filenames
    imageT1 = [y for x in imageT1 for y in x]
    imageT1c = [y for x in imageT1c for y in x]
    imageT2 = [y for x in imageT2 for y in x]
    label = [y for x in label for y in x]

    # Check for consistency
    import difflib

    for i in range(len(imageT1)):

        t1t2_seq = difflib.SequenceMatcher(None, imageT1[i].lower(), imageT2[
            i].lower())
        t1t1c_seq = difflib.SequenceMatcher(None, imageT1[i].lower(), imageT1c[
            i].lower())
        t1lab_seq = difflib.SequenceMatcher(None, imageT1[i].lower(), label[
            i].lower())

        seq = [t1t2_seq.ratio(), t1t1c_seq.ratio(), t1lab_seq.ratio()]
        seqbool = [i < 0.85 for i in seq]
        if any(seqbool):
            print('File mismatch found!')
            print('Ratio: ', seq)
            print('%s\t%s\t%s\t%s' % (os.path.split(imageT1[i])[1],
                                      os.path.split(imageT1c[i])[1],
                                      os.path.split(imageT2[i])[1],
                                      os.path.split(label[i])[1]))

        filenames = []
        for i in range(len(imageT1)):

            filenames.append([imageT1[i], imageT1c[i], imageT2[i], label[i]])

        print('%d files found!' % len(filenames))

        return filenames


def load_filenames_2nd(base_path, seg_list='/home/matt/Documents/SegSarcoma/anna_segmentations.txt'):

    # Load first dataset - Stephanie
    # Get subdirs
    subdirs = get_foldernames(base_path)

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

    # Concatenate lists
    imageT1 = image1T1 + image2T1
    imageT1c = image1T1c + image2T1c
    imageT2 = image1T2 + image2T2
    label = label1 + label2

    # Flatten the list of filenames
    imageT1 = [y for x in imageT1 for y in x]
    imageT1c = [y for x in imageT1c for y in x]
    imageT2 = [y for x in imageT2 for y in x]
    label = [y for x in label for y in x]

    # Initialize and append to the list of filenames
    filenames = []
    for ii in range(len(imageT1)):

        filenames.append([imageT1[ii], imageT1c[ii], imageT2[ii], label[ii]])

    print('First batch of segmentations\n\t%d files found!' % len(filenames))


    ######### Load 2nd dataset - Ana #########

    # Load a list of fixed segmentation files
    fname = seg_list
    with open(fname, 'r') as f:
        seg_files = f.readlines()

    # Clean filenames
    seg_files = [i.strip() for i in seg_files]

    print('Second batch of segmentations\n\t{} files found!'.format(len(seg_files)))

    # Loop over segmentations
    for seg_file in seg_files:

        # Split the path name
        path, _ = os.path.split(seg_file)

        # Get image names
        imageT1 = glob(os.path.join(path, 'T1.nii*'))[0]
        imageT1c = glob(os.path.join(path, 'T1c.nii*'))[0]
        imageT2 = glob(os.path.join(path, 'T2_cor.nii*'))[0]

        # Append names to the list of files
        filenames.append([imageT1, imageT1c, imageT2, seg_file])

    print('Total: %d files!' % len(filenames))

    return filenames


def load_data(filenames, block_size, oversamp, lab_trun, adptive_hist=False):
    # print('Loading datasets')
    X = np.empty(shape=(0, 0, 0, 0, 0))
    for file in tqdm(filenames, ncols=100):

        tmpx = load_data_3D(file[:-1])
        sz = tmpx.shape

        tmpy = load_label_3D(file[-1], sz)

        # import matplotlib
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.imshow(np.concatenate((tmpx[35, :, :, 0].T, tmpy[35, :, :, 0].T), axis=1), cmap='gray')
        # plt.show()

        # If the image dimensions of the label and inputs match proceed
        if tmpx.shape[:3] == tmpy.shape[:3]:

            # Crop the data
            mask = np.zeros(shape=(sz[1]), dtype=bool)
            mask[30:-30] = True
            tmpx = tmpx[:, :, mask, :]
            tmpy = tmpy[:, :, mask, :]
            orig_size = tmpx.shape

            # Adaptive histogram equalization
            if adptive_hist:
                tmpx = adaptive_hist_eq_3D(tmpx, patch_size=[20, 20, 20])

            tmpx, tmpy = make_train_3D(tmpx, tmpy, block_size=block_size,
                                 oversamp=oversamp,
                                 lab_trun=lab_trun)

            try:
                X = np.concatenate((X, tmpx), axis=0)
                Y = np.concatenate((Y, tmpy), axis=0)
            except ValueError:
                X = tmpx
                Y = tmpy

    return X, Y, orig_size


def keep_t2(filenames):
    """
    Keeps only filenames which pertain to T2 weighted images and labels
    Args:
        filenames (list): a list of a list of filenames. T2 images are found
            in indexes filenames[i][2].

    Returns:
        list of list: a list with len(filenames). Each element is a list of
            two strings, T2 and label images.

    """

    files = []

    for file in filenames:

        files.append(file[2:])

    return files


if __name__ == "__main__":

    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'

    filenames = load_filenames_2nd(base_path=image_base_path)

    epochs = 600
    batch_size = 20
    block_size = [18, 142, 142]
    oversamp = 1.0
    oversamp_test = 1.0
    lab_trun = 2
    im_freq = 50
    val_split = 0.2
    test_split = 0.1
    lr = 2e-4
    X, Y, orig_sz = load_data(filenames, block_size, oversamp, lab_trun)
