import os
from glob2 import glob
import nibabel as nib
import numpy as np
import keras
import tensorflow as tf
import itertools
from time import time
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import color, measure


# Set up plotting properties
sns.set(style='ticks', palette='Spectral', font_scale=1.5)
rcParams['figure.figsize'] = 6, 4


def load_data(filenames, block_size, oversamp, lab_trun, adptive_hist=False):

    X = np.empty(shape=(0, 0, 0, 0, 0))
    file = filenames

    tmpx = load_data_3D(file)
    sz = tmpx.shape

    # tmpy = load_label_3D(file[-1], sz)
    tmpy = np.zeros(shape=(sz[0], sz[1], sz[2], 1))

    # If the image dimensions of the label and inputs match proceed
    if tmpx.shape[:3] == tmpy.shape[:3]:

        # Crop the data
        # mask = np.zeros(shape=(sz[1]), dtype=bool)
        # mask[30:-30] = True
        # tmpx = tmpx[:, :, mask, :]
        # tmpy = tmpy[:, :, mask, :]
        orig_size = tmpx.shape

        # Make patches
        tmpx, tmpy = make_train_3D(tmpx, tmpy, block_size=block_size,
                             oversamp=oversamp,
                             lab_trun=lab_trun)

        try:
            X = np.concatenate((X, tmpx), axis=0)
            # Y = np.concatenate((Y, tmpy), axis=0)
        except ValueError:
            X = tmpx
            # Y = tmpy

    return X, orig_size


def load_data_3D(files):
    images = np.array([nib.load(file).get_data().astype('float32').squeeze() for file in files])

    images = np.swapaxes(images, 0, 3)

    # Normalize images
    sz = images.shape
    for z in range(sz[3]):
        mn = images[:, :, :, z].mean()
        std = images[:, :, :, z].std()
        images[:, :, :, z] -= mn
        images[:, :, :, z] /= std

    return images


def make_train_3D(X, Y, block_size, oversamp=1, lab_trun=4):
    print('Making test volume patches')
    sz = X.shape  # z, x, y, channels, vols

    if len(sz) == 4:
        sz = sz + (1,)
        X = X[:, :, :, :, np.newaxis]
        Y = Y[:, :, :, :, np.newaxis]

    # Number of volumes in each dimension
    num_vols = [int((sz[i] * oversamp) // (block_size[i] - lab_trun) + 1) for i in range(len(block_size))]

    # Get starting indices of each block (zind[0}:zind[0] + block_size[0])
    zind = np.linspace(0, sz[0] - block_size[0], num_vols[0], dtype=int)
    xind = np.linspace(0, sz[1] - block_size[1], num_vols[1], dtype=int)
    yind = np.linspace(0, sz[2] - block_size[2], num_vols[2], dtype=int)

    # Preallocate volumes - samples, block, block, block, channels
    in_patches = np.zeros(shape=(np.product(num_vols) * sz[4], block_size[0], block_size[1], block_size[2], sz[3]))
    lab_patches = np.zeros(shape=(np.product(num_vols) * sz[4], block_size[0] - lab_trun, block_size[1] - lab_trun, block_size[2] - lab_trun, 1))

    # Make volumes
    n = 0
    for vol in range(sz[4]):
        for z in itertools.product(zind, xind, yind):
            # print(z)
            in_patches[n, :, :, :, :] = X[z[0]:z[0] + block_size[0], z[1]:z[1] + block_size[1], z[2]:z[2] + block_size[2], :, vol]

            lab_patches[n, :, :, :, :] = Y[z[0] + lab_trun//2:z[0] + block_size[0] - lab_trun//2, z[1] + lab_trun//2:z[1] + block_size[1] - lab_trun//2, z[2] + lab_trun//2:z[2] + block_size[2] - lab_trun//2, :, vol]

            n += 1

    return in_patches, lab_patches


def load_models(path):
    """
    Loads a list of models
    Args:
        paths (list): list of paths to models (not including the filename)

    Returns:

    """


    model_name = os.path.join(path, 'Trained_model.h5')

    model = keras.models.load_model(model_name,
                                         custom_objects=
                                         {'dice_loss': dice_loss,
                                          'dice_metric': dice_metric})

    return model


def seg_from_model(model_path, im_paths, threshold):
    """

    Args:
        model_path (str): path to trained model (path only)
        im_paths (list of lists of 3 strings): paths to three contrast images

    Returns:

    """

    # Set up data constants
    block_size = [18, 142, 142]
    oversamp_test = 2.0
    lab_trun = 2

    # Load models
    model = load_models(model_path)

    # Load data
    x, orig_size = load_data(im_paths, block_size, oversamp_test, lab_trun)

    # Make predictions
    t = time()
    with tf.device('GPU:1'):
        y_pred = model.predict(x, batch_size=20)
    print('Time to run segmentations: %0.3f seconds' % (time() - t))

    # Reconstruct images
    x, y_pred = recon_test_3D(X=x, Y=y_pred, orig_size=orig_size, block_size=block_size, oversamp=oversamp_test,
                              lab_trun=lab_trun)

    # Swap axes
    y_pred = np.rollaxis(y_pred, 0, 2).swapaxes(1, 2)
    x =  np.rollaxis(x, 0, 2).swapaxes(1, 2)

    # Threshold segmentation
    y_thresh = y_pred > threshold

    # Remove all but the tumor label
    y_thresh = remove_extra_labels(y_thresh)

    return x[:, :, :, -1, -1], y_thresh.astype(np.single)


def recon_test_3D(X, Y, orig_size, block_size, oversamp=1, lab_trun=4):
    print('Reconstructing test volume from patches')

    # Add volume dimension if it does not exist
    if len(orig_size) == 4:
        orig_size = orig_size + (1,)

    # Add volume axis if it does not exist
    sz = X.shape
    if len(sz) < 6:
        X = X[:, :, :, :, :, np.newaxis]
        Y = Y[:, :, :, :, :, np.newaxis]



    # Number of volumes in each dimension
    num_vols = [int((orig_size[i] * oversamp) // (block_size[i] - lab_trun) + 1) for i in range(len(block_size))]

    # Get starting indices of each block (zind[0}:zind[0] + block_size[0])
    zind = np.linspace(0, orig_size[0] - block_size[0], num_vols[0], dtype=int)
    xind = np.linspace(0, orig_size[1] - block_size[1], num_vols[1], dtype=int)
    yind = np.linspace(0, orig_size[2] - block_size[2], num_vols[2], dtype=int)

    # Preallocate arrays - z, x, y, channels, vols
    in_recon = np.zeros(shape=(orig_size))
    lab_recon = np.zeros(shape=(orig_size[0], orig_size[1], orig_size[2], 1, 1))
    inds_in = np.zeros_like(in_recon, dtype=np.int8)
    inds_lab = np.zeros_like(lab_recon, dtype=np.int8)

    # Reconstruct images

    print('orig_size', orig_size)
    for vol in range(orig_size[4]):
        n = 0
        for z in itertools.product(zind, xind, yind):
            # print(z)
            # Update images
            in_recon[z[0]:z[0] + block_size[0], z[1]:z[1] + block_size[1], z[2]:z[2] + block_size[2], :, vol] += X[n, :, :, :, :, vol]
            lab_recon[z[0] + lab_trun//2:z[0] + block_size[0] - lab_trun//2, z[1] + lab_trun//2:z[1] + block_size[1] - lab_trun//2, z[2] + lab_trun//2:z[2] + block_size[2] - lab_trun//2, :, vol] += Y[n, :, :, :, :, vol]

            # Keep track of duplicate values
            inds_in[z[0]:z[0] + block_size[0], z[1]:z[1] + block_size[1], z[2]:z[2] + block_size[2], :, vol] += 1
            inds_lab[z[0] + lab_trun//2:z[0] + block_size[0] - lab_trun//2, z[1] + lab_trun//2:z[1] + block_size[1] - lab_trun//2, z[2] + lab_trun//2:z[2] + block_size[2] - lab_trun//2, :, vol] += 1

            n += 1

    in_recon /= inds_in
    lab_recon[inds_lab > 0] /= inds_lab[inds_lab > 0]

    return in_recon, lab_recon


def dice_loss(y_true, y_pred):
    threshold = 0.5
    smooth = 1e-5

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)
    # y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    # y_true = tf.cast(y_true > threshold, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    # new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)

    hard_dice = 1 - tf.reduce_mean(hard_dice)

    return hard_dice


def dice_metric(y_true, y_pred):

    threshold = 0.5

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)
    # y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    # y_true = tf.cast(y_true > threshold, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    # new haodong
    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice


def display_segmentations(t2, y_pred, save_path, sname='segs.png'):
    """
    Saves segmentations as an overlayed montage.
    Args:
        t2 (3D numpy array): T2 image
        y_pred (3D numpy array): Binary segmentation
        save_path (str): directory in which to save images

    Returns:

    """

    # Make mask see-through
    y_mask = y_pred.squeeze()

    # Set up slices to plot
    ims = 4
    slices = range(0, y_pred.shape[2], ims)
    rows = 2
    cols = len(slices) // rows
    resh = (y_pred.shape[0] * rows, y_pred.shape[1] * cols)

    t2_im = np.zeros(shape=(resh))
    y_mask_im = np.zeros_like(t2_im)
    r = [0, y_pred.shape[0]]
    c = [0, y_pred.shape[0]]
    n = 0
    for i in range(rows):
        c = [0, y_pred.shape[0]]

        for ii in range(cols):
            t2_im[r[0]:r[1], c[0]:c[1]] = t2[:, :, slices[n]].T
            y_mask_im[r[0]:r[1], c[0]:c[1]] = y_mask[:, :, slices[n]].T
            n += 1
            c = [i + y_pred.shape[0] for i in c]

        r = [i + y_pred.shape[0] for i in r]

    # Mask out background
    y_mask_im = np.ma.masked_where(y_mask_im.astype(bool) == 0, y_mask_im.astype(bool))

    # Save images
    # https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
    # Convert images to RGB
    t2_im_rep = np.dstack((t2_im, t2_im, t2_im))
    y_mask_im_rep = np.zeros_like(t2_im_rep)
    y_mask_im_rep[:, :, 1] = y_mask_im

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    y_mask_im_hsv = color.rgb2hsv(y_mask_im_rep)
    t2_im_hsv = color.rgb2hsv(t2_im_rep)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    alpha = 0.6
    t2_im_hsv[:, :, 0] = y_mask_im_hsv[:, :, 0]
    t2_im_hsv[:, :, 1] = y_mask_im_hsv[:, :, 1] * alpha;

    # Convert bach to RGB
    im_masked = color.hsv2rgb(t2_im_hsv)

    # Convert image to 8-bit
    im_masked -= im_masked.min()
    im_masked /= im_masked.max() * 0.8
    im_masked *= 255
    im_masked = im_masked.astype(np.uint8)

    # Save as image using PIL
    im = Image.fromarray(im_masked)
    sname = os.path.join(save_path, sname)
    im.save(sname, 'png')


def remove_extra_labels(y_pred):
    """
    Finds the largest continous region in the tumor label. All other regions are discarded.
    Args:
        y_pred (3D numpy array): thresholded network ouput

    Returns:
        (3D numpy array): returned segmentation
    """

    # Convert predictions to integer mask
    y_mask = y_pred.astype(np.uint8).squeeze()

    # Find continous regions
    labels = measure.label(y_mask, connectivity=3)

    # Find the number of counts for each region
    vals, counts = np.unique(labels, return_counts=True)

    # Remove background
    bg = labels[0, 0, 0]
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    # Get the largest labels
    ind = np.argmax(counts)
    tumor_val = vals[ind]

    # Return clean predictions
    y_out = (labels == tumor_val).astype(np.float)

    return y_out


if __name__ == '__main__':

    # No skip network
    paths = ['E:/MR Data/ML_Results/2019_01_17_20-26-27_onlyT2_lr2e-4_1000ep',
             'E:/MR Data/ML_Results/2019_01_18_08-58-04_all_contrasts_lr2e-4_1000ep'
             ]
    spath = 'E:/MR Data/ML_Results/SPIE/NoSkip'
    thresholds = [0.958, 0.812]
    seg_from_model(paths, spath, thresholds)

    # Skip network
    paths = ['W:/Matt/ML_Sarcoma_Results/2019_01_21_16-38-29_skip_onlyT2_lr2e-4_400ep',
             'W:/Matt/ML_Sarcoma_Results/2019_01_22_01-33-59_skip_all_contrasts_lr2e-4_400ep'
             ]
    spath = 'E:/MR Data/ML_Results/SPIE/Skip'
    thresholds = [0.958, 0.812]
    seg_from_model(paths, spath, thresholds)
