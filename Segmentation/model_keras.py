"""
A set of functions defining segmentation models and code for evaluating networks.
"""
import os
import numpy as np
import tensorflow as tf
import keras
import nibabel as nib
import nrrd
import itertools
import json
from skimage.transform import resize
from skimage import measure
from scipy.ndimage.morphology import binary_closing
import matplotlib.pyplot as plt
from shutil import copy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, \
    f1_score, roc_curve, roc_auc_score, jaccard_similarity_score


def cnn_model_3D_3lyr_relu_dice(pretrained_weights=None, input_size=(25, 25, 25, 4), lr=1e-5):
    padding = 'VALID'
    """ CNN model function"""
    # Constants
    chan_in = 4 #features.shape[4]
    in_shape = input_size #(25, 25, 25, 4)
    f1 = 24
    f2 = 48
    f3 = 64
    f4 = 64
    f5 = 48
    f6 = 24
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    strides = (1, 1, 1)


    with tf.device('/GPU:0'):
        # Input layer
        input_layer = keras.layers.Input(shape=in_shape)
        drop1 = keras.layers.Dropout(0.1)(input_layer)

        # Convolutional layers
        conv1 = keras.layers.Conv3D(filters=f1, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv1')(drop1)
        pool1 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool1')(conv1)


        conv2 = keras.layers.Conv3D(filters=f2, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv2')(pool1)
        pool2 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool2')(conv2)

        conv3 = keras.layers.Conv3D(filters=f3, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv3')(pool2)

        # Deconvolutional layers
        dconv1 = keras.layers.Deconvolution3D(filters=f4, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv1')(conv3)
        unpool1 = keras.layers.UpSampling3D(size=pool_size)(dconv1)

        dconv2 = keras.layers.Conv3DTranspose(filters=f5, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv2')(unpool1)
        unpool2 = keras.layers.UpSampling3D(size=pool_size)(dconv2)

        dconv3 = keras.layers.Conv3DTranspose(filters=f6, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv3')(unpool2)

        output = keras.layers.Conv3D(filters=1, kernel_size=kernel_size, padding=padding, activation='sigmoid')(dconv3)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=opt, loss=dice_loss, metrics=[tf.keras.metrics.binary_accuracy, dice_metric])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, opt

def cnn_model_3D_3lyr_do_relu_dice(pretrained_weights=None, input_size=(25, 25, 25, 4), lr=1e-5):
    padding = 'VALID'
    """ CNN model function"""
    # Constants
    chan_in = 4 #features.shape[4]
    in_shape = input_size #(25, 25, 25, 4)
    f1 = 24
    f2 = 48
    f3 = 64
    f4 = 64
    f5 = 48
    f6 = 24
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    strides = (1, 1, 1)


    with tf.device('/GPU:0'):
        # Input layer
        input_layer = keras.layers.Input(shape=in_shape)
        drop0 = keras.layers.SpatialDropout3D(0.1)(input_layer)

        # Convolutional layers
        conv1 = keras.layers.Conv3D(filters=f1, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv1')(drop0)
        drop1 = keras.layers.SpatialDropout3D(0.1)(conv1)
        pool1 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool1')(drop1)


        conv2 = keras.layers.Conv3D(filters=f2, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv2')(pool1)
        drop2 = keras.layers.SpatialDropout3D(0.1)(conv2)
        pool2 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool2')(drop2)

        conv3 = keras.layers.Conv3D(filters=f3, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv3')(pool2)
        drop3 = keras.layers.SpatialDropout3D(0.1)(conv3)

        # Deconvolutional layers
        dconv1 = keras.layers.Deconvolution3D(filters=f4, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv1')(drop3)
        drop3 = keras.layers.SpatialDropout3D(0.1)(dconv1)
        unpool1 = keras.layers.UpSampling3D(size=pool_size)(drop3)

        dconv2 = keras.layers.Conv3DTranspose(filters=f5, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv2')(unpool1)
        drop4 = keras.layers.SpatialDropout3D(0.1)(dconv2)
        unpool2 = keras.layers.UpSampling3D(size=pool_size)(drop4)

        dconv3 = keras.layers.Conv3DTranspose(filters=f6, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv3')(unpool2)

        output = keras.layers.Conv3D(filters=1, kernel_size=kernel_size, padding=padding, activation='sigmoid')(dconv3)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=opt, loss=dice_loss, metrics=['accuracy'])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, opt

def cnn_model_3D_3lyr_do_relu_dice_skip(pretrained_weights=None, input_size=(25, 25, 25, 4), lr=1e-5):
    padding = 'VALID'
    """ CNN model function"""
    # Constants
    chan_in = 4 #features.shape[4]
    in_shape = input_size #(25, 25, 25, 4)
    f1 = 24
    f2 = 48
    f3 = 64
    f4 = 64
    f5 = 48
    f6 = 24
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    strides = (1, 1, 1)


    with tf.device('/GPU:0'):
        # Input layer
        input_layer = keras.layers.Input(shape=in_shape)
        drop0 = keras.layers.SpatialDropout3D(0.1)(input_layer)

        # Convolutional layers
        conv1 = keras.layers.Conv3D(filters=f1, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv1')(drop0)
        drop1 = keras.layers.SpatialDropout3D(0.1)(conv1)
        pool1 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool1')(drop1)


        conv2 = keras.layers.Conv3D(filters=f2, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv2')(pool1)
        drop2 = keras.layers.SpatialDropout3D(0.1)(conv2)
        pool2 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool2')(drop2)

        conv3 = keras.layers.Conv3D(filters=f3, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv3')(pool2)
        drop3 = keras.layers.SpatialDropout3D(0.1)(conv3)

        # Deconvolutional layers
        dconv1 = keras.layers.Deconvolution3D(filters=f4, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv1')(drop3)
        drop3 = keras.layers.SpatialDropout3D(0.1)(dconv1)
        unpool1 = keras.layers.UpSampling3D(size=pool_size)(drop3)

        cat1 = keras.layers.concatenate([drop2, unpool1])
        dconv2 = keras.layers.Conv3DTranspose(filters=f5,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              activation='relu',
                                              name='Dconv2')(cat1)
        drop4 = keras.layers.SpatialDropout3D(0.1)(dconv2)
        unpool2 = keras.layers.UpSampling3D(size=pool_size)(drop4)

        cat2 = keras.layers.concatenate([drop1, unpool2])
        dconv3 = keras.layers.Conv3DTranspose(filters=f6,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              activation='relu',
                                              name='Dconv3')(cat2)

        output = keras.layers.Conv3D(filters=1, kernel_size=kernel_size, padding=padding, activation='sigmoid')(dconv3)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=opt, loss=dice_loss, metrics=[keras.metrics.binary_accuracy, dice_metric])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, opt

def cnn_model_3D_3lyr_do_relu_xentropy_skip(pretrained_weights=None, input_size=(25, 25, 25, 4), lr=1e-5):
    padding = 'VALID'
    """ CNN model function"""
    # Constants
    chan_in = 4 #features.shape[4]
    in_shape = input_size #(25, 25, 25, 4)
    f1 = 24
    f2 = 48
    f3 = 64
    f4 = 64
    f5 = 48
    f6 = 24
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    strides = (1, 1, 1)


    with tf.device('/GPU:0'):
        # Input layer
        input_layer = keras.layers.Input(shape=in_shape)
        drop0 = keras.layers.SpatialDropout3D(0.1)(input_layer)

        # Convolutional layers
        conv1 = keras.layers.Conv3D(filters=f1, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv1')(drop0)
        drop1 = keras.layers.SpatialDropout3D(0.1)(conv1)
        pool1 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool1')(drop1)


        conv2 = keras.layers.Conv3D(filters=f2, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv2')(pool1)
        drop2 = keras.layers.SpatialDropout3D(0.1)(conv2)
        pool2 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool2')(drop2)

        conv3 = keras.layers.Conv3D(filters=f3, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv3')(pool2)
        drop3 = keras.layers.SpatialDropout3D(0.1)(conv3)

        # Deconvolutional layers
        dconv1 = keras.layers.Deconvolution3D(filters=f4, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv1')(drop3)
        drop3 = keras.layers.SpatialDropout3D(0.1)(dconv1)
        unpool1 = keras.layers.UpSampling3D(size=pool_size)(drop3)

        cat1 = keras.layers.concatenate([drop2, unpool1])
        dconv2 = keras.layers.Conv3DTranspose(filters=f5,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              activation='relu',
                                              name='Dconv2')(cat1)
        drop4 = keras.layers.SpatialDropout3D(0.1)(dconv2)
        unpool2 = keras.layers.UpSampling3D(size=pool_size)(drop4)

        cat2 = keras.layers.concatenate([drop1, unpool2])
        dconv3 = keras.layers.Conv3DTranspose(filters=f6,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              activation='relu',
                                              name='Dconv3')(cat2)

        output = keras.layers.Conv3D(filters=1, kernel_size=kernel_size, padding=padding, activation='sigmoid')(dconv3)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, opt

def cnn_model_3D_3lyr_do_relu_xentropy(pretrained_weights=None, input_size=(25, 25, 25, 4), lr=1e-5):
    padding = 'VALID'
    """ CNN model function"""
    # Constants
    chan_in = 4 #features.shape[4]
    in_shape = input_size #(25, 25, 25, 4)
    f1 = 24
    f2 = 48
    f3 = 64
    f4 = 64
    f5 = 48
    f6 = 24
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    strides = (1, 1, 1)



    with tf.device('/GPU:0'):
        # Input layer
        input_layer = keras.layers.Input(shape=in_shape)
        drop0 = keras.layers.Dropout(0.1)(input_layer)

        # Convolutional layers
        conv1 = keras.layers.Conv3D(filters=f1, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv1')(drop0)
        drop1 = keras.layers.Dropout(0.1)(conv1)
        pool1 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool1')(drop1)


        conv2 = keras.layers.Conv3D(filters=f2, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv2')(pool1)
        drop2 = keras.layers.Dropout(0.1)(conv2)
        pool2 = keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, data_format='channels_last', name='Pool2')(drop2)

        conv3 = keras.layers.Conv3D(filters=f3, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Conv3')(pool2)
        drop3 = keras.layers.Dropout(0.1)(conv3)

        # Deconvolutional layers
        dconv1 = keras.layers.Deconvolution3D(filters=f4, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv1')(drop3)
        drop3 = keras.layers.Dropout(0.1)(dconv1)
        unpool1 = keras.layers.UpSampling3D(size=pool_size)(drop3)

        dconv2 = keras.layers.Conv3DTranspose(filters=f5, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv2')(unpool1)
        drop4 = keras.layers.Dropout(0.1)(dconv2)
        unpool2 = keras.layers.UpSampling3D(size=pool_size)(drop4)

        dconv3 = keras.layers.Conv3DTranspose(filters=f6, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu', name='Dconv3')(unpool2)

        output = keras.layers.Conv3D(filters=1, kernel_size=kernel_size, padding=padding, activation='sigmoid')(dconv3)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[keras.metrics.binary_accuracy, dice_metric])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model, opt


def multi_gpu_model(model, opt, loss, gpus):
    parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return parallel_model


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


def load_label_3D(files, sz):
    # Load image
    if '.nii' in files:
        # If the image is Nifti format
        label = nib.load(files).get_data().astype('float32').squeeze()

        # Resize image
        label /= label.max()
        label = resize(label, (sz[1], sz[2]))

        # Make binary
        label[label < 0.5] = 0
        label[label > 0.5] = 1

        # Close the mask
        label = binary_closing(label, structure=np.ones(shape=(3, 3, 3)))

    else:
        # If the image is nrrd format
        _nrrd = nrrd.read(files)
        tmpy = _nrrd[0][0, :, :, :]
        info = _nrrd[1]

         # Swap axis to match images
        # tmpy = np.rollaxis(tmpy, axis=2)
        tmpy = np.swapaxes(tmpy, 0, 2)

        # Pad label to match images
        offsets = info['space origin'].astype(int)
        offsets_inds = [2, 1, 0]

        # Z direction
        offset = offsets[offsets_inds[0]]
        pad = sz[0] - tmpy.shape[0] - offset
        tmpy = np.concatenate((np.zeros((offset, tmpy.shape[1], tmpy.shape[2])), tmpy), axis=0)
        tmpy = np.concatenate((tmpy, np.zeros((pad, tmpy.shape[1], tmpy.shape[2]))), axis=0)

        # X direction
        offset = offsets[offsets_inds[1]]
        pad = sz[1] - tmpy.shape[1] - offset
        tmpy = np.concatenate((np.zeros((tmpy.shape[0], offset, tmpy.shape[2])), tmpy), axis=1)
        tmpy = np.concatenate((tmpy, np.zeros((tmpy.shape[0], pad, tmpy.shape[2]))), axis=1)

        # Y direction
        offset = offsets[offsets_inds[2]]
        pad = sz[2] - tmpy.shape[2] - offset
        tmpy = np.concatenate((np.zeros((tmpy.shape[0], tmpy.shape[1], offset)), tmpy), axis=2)
        tmpy = np.concatenate((tmpy, np.zeros((tmpy.shape[0], tmpy.shape[1], pad))), axis=2)

        # Assign to label
        label = np.swapaxes(tmpy, 0, 2).astype(np.bool)

    # Reorganize data to put slices first
    label = np.swapaxes(label[np.newaxis, :, :, :], 0, 3)

    return label


def make_train_3D(X, Y, block_size, oversamp=1, lab_trun=4):
    # print('Making volume patches')
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


def recon_test_3D(X, Y, orig_size, block_size, oversamp=1, lab_trun=4):
    print('Reconstructing volume from patches')

    # Number of volumes in each dimension
    num_vols = [int((orig_size[i] * oversamp) // (block_size[i] - lab_trun) + 1) for i in range(len(block_size))]

    # Get input dimensions
    szy = Y.shape
    sz = X.shape

    # Add volume dimension if it does not exist
    total_vols = int(X.shape[0] / np.prod(num_vols))
    if len(orig_size) == 4:
        orig_size = orig_size + (total_vols,)

    if total_vols > 1:

        split = int(szy[0] / total_vols)
        y_vols = np.zeros(shape=(split, szy[1], szy[2], szy[3], szy[4], total_vols) )
        x_vols = np.zeros(shape=(split, sz[1], sz[2], sz[3], sz[4], total_vols))

        for i in range(total_vols):
            inds = list(range(i*split, (i+1)*split))
            y_vols[:,:,:,:,:,i] = Y[inds,:,:,:,:]
            x_vols[:,:,:,:,:,i] = X[inds,:,:,:,:]

        # Update volumes
        Y = y_vols
        X = x_vols

        # Clear temp variables
        y_vols, x_vols = None, None

    # Add volume axis if it does not exist
    szy = Y.shape
    sz = X.shape
    if len(sz) < 6:
        X = X[:, :, :, :, :, np.newaxis]
        Y = Y[:, :, :, :, :, np.newaxis]

    # Get starting indices of each block (zind[0}:zind[0] + block_size[0])
    zind = np.linspace(0, orig_size[0] - block_size[0], num_vols[0], dtype=int)
    xind = np.linspace(0, orig_size[1] - block_size[1], num_vols[1], dtype=int)
    yind = np.linspace(0, orig_size[2] - block_size[2], num_vols[2], dtype=int)

    # Preallocate arrays - z, x, y, channels, vols
    in_recon = np.zeros(shape=(orig_size))
    lab_recon = np.zeros(shape=(orig_size[0], orig_size[1], orig_size[2], 1, orig_size[4]))
    inds_in = np.zeros_like(in_recon, dtype=np.int8)
    inds_lab = np.zeros_like(lab_recon, dtype=np.int8)

    # Reconstruct images
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


def training_eval(model, features, spath, orig_size, oversamp, lab_trun):
    # Eval training data
    train_test = features['X']
    train_test_lab = features['Y']

    # Get block size
    block_size = train_test.shape
    block_size = block_size[1:4]

    train_ev = model.evaluate(x=train_test, y=train_test_lab)
    print('Training evaluation\n\t')
    print(train_ev)
    train_pred = model.predict(train_test)

    # Compute ideal threshold
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true=train_test_lab.reshape(-1),
                                                             probas_pred=train_pred.reshape(-1), pos_label=None)
    sub = np.abs(recalls[:-1] - precisions[:-1])
    # val, idx = min((val, idx) for (idx, val) in enumerate(sub))
    # threshold = thresholds[idx]
    inds = sub < 0.025
    ind = np.argmax(inds)
    threshold = thresholds[ind]
    # threshold = threshold[0]


    # Use threshold
    train_pred = train_pred > threshold

    # Convert back to images
    _, Y = recon_test_3D(X=train_test, Y=train_test_lab, orig_size=orig_size, block_size=block_size, oversamp=oversamp,
                         lab_trun=lab_trun)
    X, train_pred = recon_test_3D(X=train_test, Y=train_pred, orig_size=orig_size, block_size=block_size,
                                  oversamp=oversamp, lab_trun=lab_trun)

    # Plot images
    Y_mask = np.ma.masked_where(Y.astype(bool) == 0, Y.astype(bool))
    train_pred_mask = np.ma.masked_where(train_pred.astype(bool) == 0, train_pred.astype(bool))
    # train_pred_mask[train_pred_mask<0.9] = 0

    slices = range(20, 40, 4)
    resh = (Y.shape[1] * len(slices), Y.shape[2])
    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(train_pred_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)
    ax[2].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
    ax[2].imshow(Y_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)

    ax[0].set_title('Input slice (T2)')
    ax[1].set_title('Output slice')
    ax[2].set_title('Label')

    ax[2].set_xticklabels(slices)

    for a in ax.ravel():
        a.axis('off')

    fig.savefig(os.path.join(spath, 'Train_set.pdf'), dpi=300, format='pdf', frameon=False)


def test_set_3D(model, X, Y, spath, orig_size, block_size, oversamp, lab_trun, batch_size, threshold=None, vols=None, continuous=False):

    sz = X.shape
    szy = Y.shape
    Y_pred = np.zeros_like(Y)

    # Test the segemntation
    if len(sz) > 5:
        vols = sz[5]
        for vol in range(vols):
            temp = X[:, :, :, :, :, vol]
            Y_pred[:, :, :, :, :, vol] = model.predict(x=temp, batch_size=batch_size)
    else:

        Y_pred = model.predict(x=X, batch_size=batch_size)

    # Evaluate metrics
    if threshold:
        print('\tUsing threshold from training set:\t%0.3f' %threshold)

    else:
        threshold = 0.9
        print('\tUsing fixed threshold:\t%0.3f' %threshold)

    Y_pred_thresh = Y_pred > threshold

    # convert volume patches to images
    _, Y             = recon_test_3D(X=X, Y=Y, orig_size=orig_size, block_size=block_size, oversamp=oversamp, lab_trun=lab_trun)
    _, Y_pred        = recon_test_3D(X=X, Y=Y_pred, orig_size=orig_size, block_size=block_size, oversamp=oversamp, lab_trun=lab_trun)
    X, Y_pred_thresh = recon_test_3D(X=X, Y=Y_pred_thresh, orig_size=orig_size, block_size=block_size, oversamp=oversamp, lab_trun=lab_trun)

    # Account for partial volume effects
    Y_pred_thresh = Y_pred_thresh > 0.1

    _ = model_evaluation(spath=spath, Y=Y, Y_pred=Y_pred, epoch='End', images=True, threshold=threshold, continous=continuous)

    # Plot images
    y_plot = Y[:, :, :, :, 0]
    y_plot_pred = Y_pred_thresh[:, :, :, :, 0]
    Y_mask = np.ma.masked_where(y_plot.astype(bool) == 0, y_plot.astype(bool))
    test_out_mask = np.ma.masked_where(y_plot_pred.astype(bool) == 0, y_plot_pred.astype(bool))

    slices = range(20, 40, 4)
    resh = (y_plot.shape[1] * len(slices), y_plot.shape[2])
    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].imshow(X[slices, :, :, -1, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(X[slices, :, :, -1, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(test_out_mask[slices, :, :, 0].reshape(resh).T, cmap='summer', alpha=0.5)
    ax[2].imshow(X[slices, :, :, -1, 0].reshape(resh).T, cmap='gray')
    ax[2].imshow(Y_mask[slices, :, :, 0].reshape(resh).T, cmap='summer', alpha=0.5)

    ax[0].set_title('Input slice (T2)')
    ax[1].set_title('Output slice')
    ax[2].set_title('Label')

    ax[2].set_xticklabels(slices)

    for a in ax.ravel():
        a.axis('off')

    backgrnd = np.concatenate((X[slices, :, :, -1, 0].reshape(resh).T,
                               X[slices, :, :, -1, 0].reshape(resh).T,
                               X[slices, :, :, -1, 0].reshape(resh).T))

    fig.savefig(os.path.join(spath, 'Test_set.png'), dpi=300, format='png', frameon=False)

    # Rotate image axis for viewing
    X = X.swapaxes(0, 2).swapaxes(0, 1)
    Y = Y.swapaxes(0, 2).swapaxes(0, 1).astype(np.int16)
    Y_pred = Y_pred.swapaxes(0, 2).swapaxes(0, 1)
    Y_pred_thresh = Y_pred_thresh.swapaxes(0, 2).swapaxes(0, 1).astype(np.int16)

    # Save volumes
    nib.save(nib.Nifti1Image(X, np.eye(4)), os.path.join(spath, 'X.nii'))
    nib.save(nib.Nifti1Image(Y, np.eye(4)), os.path.join(spath, 'Y_lab.nii.gz'))
    nib.save(nib.Nifti1Image(Y_pred_thresh, np.eye(4)), os.path.join(spath, 'Y_pred.nii.gz'))
    nib.save(nib.Nifti1Image(Y_pred, np.eye(4)), os.path.join(spath, 'Y_pred_prob.nii.gz'))


def remove_extra_labels(y_pred):
    """
    Finds the largest continous region in the tumor label. All other regions are discarded.
    Args:
        y_pred (3D numpy array): thresholded network ouput

    Returns:
        (3D numpy array): returned segmentation
    """

    # Get sizes
    sz = y_pred.shape
    vols = sz[-1]

    # Initialize output array
    y_out = np.zeros_like(y_pred)

    for z in range(vols):

        # Convert predictions to integer mask
        y_mask = y_pred[:,:,:,:,z].astype(np.uint8).squeeze()

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
        y_mask = (labels == tumor_val).astype(np.float)

        y_out[:,:,:,0,z] = y_mask

    return y_out


def test_set_3D_kfold(model, X, Y, spath, spath_iter, orig_size, block_size, oversamp, lab_trun, batch_size, threshold=None):

    sz = X.shape
    Y_pred = np.zeros_like(Y)

    # Test the segemntation
    if len(sz) > 5:
        vols = sz[5]
        for vol in range(vols):
            temp = X[:, :, :, :, :, vol]
            Y_pred[:, :, :, :, :, vol] = model.predict(x=temp, batch_size=batch_size)
    else:

        Y_pred = model.predict(x=X, batch_size=batch_size)



    # Evaluate metrics
    if threshold:
        print('\tUsing threshold from training set:\t%0.3f' %threshold)

    else:
        threshold = 0.9
        print('\tUsing fixed threshold:\t%0.3f' %threshold)


    # _ = model_evaluation(spath=spath_iter, Y=Y, Y_pred=Y_pred, epoch='End', images=True, threshold=threshold)
    _ = model_evaluation_kfold(spath=spath, Y=Y, Y_pred=Y_pred, epoch='End', images=False, threshold=threshold)


    Y_pred_thresh = Y_pred > threshold

    # convert volume patches to images
    _, Y = recon_test_3D(X=X, Y=Y, orig_size=orig_size, block_size=block_size, oversamp=oversamp, lab_trun=lab_trun)
    _, Y_pred_thresh = recon_test_3D(X=X, Y=Y_pred_thresh, orig_size=orig_size, block_size=block_size, oversamp=oversamp,
                                       lab_trun=lab_trun)
    X, Y_pred = recon_test_3D(X=X, Y=Y_pred, orig_size=orig_size, block_size=block_size, oversamp=oversamp,
                                       lab_trun=lab_trun)

    # Convert to boolean
    Y = Y.astype('bool')
    Y_pred_thresh = Y_pred_thresh.astype('bool')

    # Plot images
    Y_mask = np.ma.masked_where(Y == 0, Y)
    test_out_mask = np.ma.masked_where(Y_pred_thresh == 0, Y_pred_thresh)

    slices = range(20, 40, 4)
    resh = (Y.shape[1] * len(slices), Y.shape[2])
    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(test_out_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)
    ax[2].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
    ax[2].imshow(Y_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)

    ax[0].set_title('Input slice (T2)')
    ax[1].set_title('Output slice')
    ax[2].set_title('Label')

    ax[2].set_xticklabels(slices)

    for a in ax.ravel():
        a.axis('off')

    f1 = f1_score(y_true=Y.reshape(-1), y_pred=Y_pred_thresh.reshape(-1))
    fid = open(os.path.join(spath, 'k-fold-metrics'), 'a')
    fid.write('Dice:\t%0.4f\n' %f1)
    fid.write('')

    backgrnd = np.concatenate((X[slices, :, :, 2, 0].reshape(resh).T, X[slices, :, :, 2, 0].reshape(resh).T, X[slices, :, :, 2, 0].reshape(resh).T))

    fig.savefig(os.path.join(spath, 'Test_set.png'), dpi=300, format='png', frameon=False)

    # Save volumes
    nib.save(nib.Nifti1Image(X, np.eye(4)), os.path.join(spath_iter, 'X.nii'))
    nib.save(nib.Nifti1Image(Y.astype('float32'), np.eye(4)), os.path.join(spath_iter, 'Y_lab.nii'))
    nib.save(nib.Nifti1Image(Y_pred_thresh.astype('float32'), np.eye(4)), os.path.join(spath_iter, 'Y_pred.nii'))
    nib.save(nib.Nifti1Image(Y_pred.astype('float32'), np.eye(4)), os.path.join(spath_iter, 'Y_pred_prob.nii'))


def test_set_3D_T2(model, X, Y, spath, orig_size, block_size, oversamp, lab_trun, batch_size, threshold=None):
    print('Evaluating the test set')

    sz = X.shape
    test_out = np.zeros_like(Y)

    # Test the segemntation
    print('\tMaking model predictions')
    if len(sz) > 5:
        vols = sz[5]
        for vol in range(vols):
            temp = X[:, :, :, :, :, vol]
            test_out[:, :, :, :, :, vol] = model.predict(x=temp, batch_size=batch_size)
    else:

        test_out = model.predict(x=X, batch_size=batch_size)


    if threshold:
        print('\tUsing threshold from training set:\t%0.3f' %threshold)

    else:
        threshold = 0.9
        print('\tUsing fixed threshold:\t%0.3f' %threshold)

    # Evaluate metrics
    # test_ev = model.evaluate(x=X, y=Y)
    # print('Test evaluation\n\t')
    # print(test_ev)
    _ = model_evaluation(spath=spath, Y=Y, Y_pred=test_out, epoch='End', images=True, threshold=threshold)

    test_out_thresh = test_out > threshold

    # convert volume patches to images
    _, Y = recon_test_3D(X=X, Y=Y, orig_size=orig_size, block_size=block_size, oversamp=oversamp, lab_trun=lab_trun)
    X, test_out_thresh = recon_test_3D(X=X, Y=test_out_thresh, orig_size=orig_size, block_size=block_size,
                                       oversamp=oversamp,
                                       lab_trun=lab_trun)

    # Plot images
    Y_mask = np.ma.masked_where(Y.astype(bool) == 0, Y.astype(bool))
    test_out_mask = np.ma.masked_where(test_out_thresh.astype(bool) == 0, test_out_thresh.astype(bool))

    slices = range(20, 40, 4)
    resh = (Y.shape[1] * len(slices), Y.shape[2])
    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].imshow(X[slices, :, :, 0, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(X[slices, :, :, 0, 0].reshape(resh).T, cmap='gray')
    ax[1].imshow(test_out_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)
    ax[2].imshow(X[slices, :, :, 0, 0].reshape(resh).T, cmap='gray')
    ax[2].imshow(Y_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)

    ax[0].set_title('Input slice (T2)')
    ax[1].set_title('Output slice')
    ax[2].set_title('Label')

    ax[2].set_xticklabels(slices)

    for a in ax.ravel():
        a.axis('off')

    fig.savefig(os.path.join(spath, 'Test_set.png'), dpi=300, format='png', frameon=False)


def load_test_data(block_size, oversamp, lab_trun):
    # Load test data
    # Data path
    dpath = 'E:/MR Data/SarcomaSegmentations/Mouse VI/Bias Corrections/'
    files = ['Mouse VI T1FLASH New Bias Correction.nii', 'Mouse VI T1FLASH w Con New Bias Correction.nii',
             'Mouse VI T2TurboRARE New Bias Correction.nii']
    files = [os.path.join(dpath, file) for file in files]

    label_file = 'E:/MR Data/SarcomaSegmentations/Mouse VI/50002 VI ROI Black and White Volume.nii'

    # Load data
    X_test = load_data_3D(files)
    sz = X_test.shape
    Y_test = load_label_3D(label_file, sz)

    # Convert to volumes
    X_test, Y_test = make_train_3D(X_test, Y_test, block_size=block_size, oversamp=oversamp, lab_trun=lab_trun)

    return X_test, Y_test, sz


def model_evaluation(spath, Y, Y_pred, epoch, images, threshold=None, continous=False):
    # Model evaluation

    print('\tEvaluating predictions\n')

    # Make sure epoch is a string
    if not isinstance(epoch, str):
        epoch = str(epoch)

    metric_file = open(os.path.join(spath, 'metrics2.txt'), 'w')

    # Precision, recall curve
    print('\t\tCalculating precision and recall')
    precisions, recalls, thresholds = precision_recall_curve(y_true=Y.reshape(-1), probas_pred=Y_pred.reshape(-1), pos_label=None)

    if images:
        fig = plt.figure(11)
        ax1 = fig.add_subplot(111)
        ax1.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        ax1.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend()
        plt.grid()
        plt.ylim([0, 1])

        fig.savefig(os.path.join(spath, 'Precision_recall_%s.svg' %epoch), dpi=300, format='svg', frameon=False)

        # Compute ROC curve
        fpr, tpr, rcthresh = roc_curve(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))
        roc_score = roc_auc_score(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))
        print('\t\tROC score:\t%0.4f' %roc_score)

        fig = plt.figure(21)
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')

        fig.savefig(os.path.join(spath, 'ROC_curve_%s.svg' %epoch), dpi=300, format='svg', frameon=False)

        plt.close('all')

    # Remove non-continuous regions - keep only the largest region
    if continous:
        Y_pred_inds = Y_pred > 0.15
        Y_pred_inds = remove_extra_labels(Y_pred_inds)
        Y_pred[~Y_pred_inds] = 0

    # Compute ideal threshold
    print('\t\tCalculating ideal threshold')
    if not threshold:
        sub = recalls[:-1] - precisions[:-1]
        # val, idx = min((val, idx) for (idx, val) in enumerate(sub))
        # threshold = thresholds[idx]
        if np.sum(sub < 0) == 0:
            ind = np.argmin(sub) - 1
        else:
            inds = sub < 0
            ind = np.argmax(inds)
        threshold = thresholds[ind]

    # Create and save a dictionary for metrics
    # df = {'false_positive': fpr.tolist(),
    #       'true_postive': tpr.tolist(),
    #       'roc_thresholds': rcthresh.tolist(),
    #       'precision': precisions.tolist(),
    #       'recall': recalls.tolist(),
    #       'thresholds': thresholds.tolist()}
    # metric_data = os.path.join(spath, 'metric_data.json')
    # log = json.dumps(df)
    #
    # with open(metric_data, 'w') as f:
    #     f.write(log)

    # Compute precision and recall
    print('\t\tCalculating ROC and DICE scores')
    test_out_thresh = (Y_pred > threshold)
    precision = precision_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    recall = recall_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    f1 = f1_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    voe = jaccard_similarity_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    roc = roc_auc_score(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))
    print('Ideal threshold:\t%0.4f' % threshold)
    print('Precision score:\t%0.4f' %precision)
    print('Recall score:\t\t%0.4f' %recall)
    print('DICE score:\t\t\t%0.4f' % f1)
    print('VOE score:\t\t\t%0.4f' % voe)

    metric_file.write('Epoch %s\n' %epoch)
    metric_file.write('\tBest threshold:\t\t%0.4f\n' %threshold)
    metric_file.write('\tROC score:\t\t%0.4f\n' % roc_score)
    metric_file.write('\tPrecision score:\t%0.4f\n' %precision)
    metric_file.write('\tRecall score:\t\t%0.4f\n' %recall)
    metric_file.write('\tDICE score:\t\t%0.4f\n' % f1)
    metric_file.write('\tVOE score:\t\t%0.4f\n\n' % voe)

    # Compute DICE similarity coefficient
    # tp = sum((Y.reshape(-1) == 1) * (test_out_thresh.reshape(-1) == 1))
    # fp = sum((Y.reshape(-1) == 0) * (test_out_thresh.reshape(-1) == 1))
    # fn = sum((Y.reshape(-1) == 1) * (test_out_thresh.reshape(-1) == 0))
    # tn = sum((Y.reshape(-1) == 0) * (test_out_thresh.reshape(-1) == 0))

    # DSC = 2 * tp / (2*tp + fp + fn) # Dice
    # VDR = abs(fp - fn) / (tp + fn)  # volume difference rate


    # print('Volume difference score:\t%0.4f\n' %VDR)


    # metric_file.write('Volume difference score:\t%0.4f\n' %VDR)

    metric_file.close()

    return threshold

def model_evaluation_kfold(spath, Y, Y_pred, epoch, images, threshold=None):
    # Model evaluation

    print('\tEvaluating predictions\n')

    # Make sure epoch is a string
    if not isinstance(epoch, str):
        epoch = str(epoch)

    metric_file = open(os.path.join(spath, 'metrics.txt'), 'a')

    # Precision, recall curve
    print('\t\tCalculating precision and recall')
    precisions, recalls, thresholds = precision_recall_curve(y_true=Y.reshape(-1), probas_pred=Y_pred.reshape(-1), pos_label=None)

    if images:
        fig = plt.figure(11)
        ax1 = fig.add_subplot(111)
        ax1.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        ax1.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend()
        plt.grid()
        plt.ylim([0, 1])

        fig.savefig(os.path.join(spath, 'Precision_recall_%s.png' %epoch), dpi=300, format='png', frameon=False)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))
        roc_score = roc_auc_score(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))
        print('\t\tROC score:\t%0.4f' %roc_score)

        fig = plt.figure(21)
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')

        fig.savefig(os.path.join(spath, 'ROC_curve_%s.png' %epoch), dpi=300, format='png', frameon=False)

        plt.close('all')

    # Compute ideal threshold
    print('\t\tCalculating ideal threshold')
    if not threshold:
        sub = recalls[:-1] - precisions[:-1]
        # val, idx = min((val, idx) for (idx, val) in enumerate(sub))
        # threshold = thresholds[idx]
        inds = sub < 0
        ind = np.argmax(inds)
        threshold = thresholds[ind]

    # threshold = threshold[0]

    # Compute precision and recall
    print('\t\tCalculating ROC and DICE scores')
    test_out_thresh = (Y_pred > threshold)
    precision = precision_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    recall = recall_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    f1 = f1_score(y_true=Y.reshape(-1), y_pred=test_out_thresh.reshape(-1))
    roc = roc_auc_score(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))
    print('Ideal threshold:\t%0.4f' % threshold)
    print('Precision score:\t%0.4f' %precision)
    print('Recall score:\t\t%0.4f' %recall)
    print('DICE score:\t\t\t%0.4f' % f1)

    metric_file.write('Epoch %s\n' %epoch)
    metric_file.write('\tBest threshold:\t\t%0.4f\n' %threshold)
    if images:
        metric_file.write('\tROC score:\t\t%0.4f\n' % roc_score)
    metric_file.write('\tPrecision score:\t%0.4f\n' %precision)
    metric_file.write('\tRecall score:\t\t%0.4f\n' %recall)
    metric_file.write('\tDICE score:\t\t%0.4f\n\n' % f1)

    # Compute DICE similarity coefficient
    # tp = sum((Y.reshape(-1) == 1) * (test_out_thresh.reshape(-1) == 1))
    # fp = sum((Y.reshape(-1) == 0) * (test_out_thresh.reshape(-1) == 1))
    # fn = sum((Y.reshape(-1) == 1) * (test_out_thresh.reshape(-1) == 0))
    # tn = sum((Y.reshape(-1) == 0) * (test_out_thresh.reshape(-1) == 0))

    # DSC = 2 * tp / (2*tp + fp + fn) # Dice
    # VDR = abs(fp - fn) / (tp + fn)  # volume difference rate


    # print('Volume difference score:\t%0.4f\n' %VDR)


    # metric_file.write('Volume difference score:\t%0.4f\n' %VDR)

    metric_file.close()

    return threshold

def training_threshold(model, spath, X, Y):
    print('Predicting best threshold based on training data')

    # Prediction based on training data
    print('\tRunning training data through model')
    Y_pred = model.predict(X, batch_size=10)

    # Calculating precision/recall curves
    print('\tCalculating precision/recall curves')
    precisions, recalls, thresholds = precision_recall_curve(
        y_true=Y.reshape(-1), probas_pred=Y_pred.reshape(-1), pos_label=None)

    if True:
        print('\t\tSaving precision/recall plot')
        fig = plt.figure(11)
        ax1 = fig.add_subplot(111)
        ax1.plot(thresholds, precisions[:-1], 'b--', label='Precision')
        ax1.plot(thresholds, recalls[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.legend()
        plt.grid()
        plt.ylim([0, 1])

        fig.savefig(os.path.join(spath, 'Precision_recall_training.png'),
                    dpi=300, format='png', frameon=False)

        # Compute ROC curve
        print('\tCalculating ROC curve')
        fpr, tpr, _ = roc_curve(y_true=Y.reshape(-1), y_score=Y_pred.reshape(-1))

        fig = plt.figure(21)
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')

        fig.savefig(os.path.join(spath, 'ROC_curve_training.png'), dpi=300,
                    format='png', frameon=False)

        plt.close('all')

    # Determining optimal threshold (precision == recall)
    print('\tDetermining optimal threshold (precision == recall)')
    sub = recalls[:-1] - precisions[:-1]
    inds = sub < 0
    if np.sum(inds) ==0:  # if recall and precision do not cross
        inds = np.argmin(recalls[:-1] - precisions[:-1])

    ind = np.argmax(inds)
    threshold = thresholds[ind]

    fid = open(os.path.join(spath, 'metrics.txt'), 'a')
    fid.write('Training threshold:\t%0.3f\n\n' %threshold)
    fid.close()

    return threshold


def dice_loss(y_true, y_pred):
    threshold = 0.5
    smooth = 1e-5
    # # Compute thresholds
    # thresholds = tf.range(start=0.4, limit=0.7, delta=0.05)
    # thresholds =  np.arange(0.4, 0.8, step=0.05).astype('float32')
    # tp = tf.metrics.true_positives_at_thresholds(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), thresholds)
    # fn = tf.metrics.false_negatives_at_thresholds(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), thresholds)
    # thresholds = tf.convert_to_tensor_or_sparse_tensor(thresholds.reshape(-1))
    # threshold = thresholds[ tf.cast( tf.argmax( tf.multiply(tp, fn) ), dtype=tf.int32 ) ]

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


def save_code(spath, calling_script):
    # Get current file
    current_file = os.path.realpath(__file__)

    # Copy current file to save directory
    copy(current_file, spath)

    # Copy calling file to save directory
    copy(calling_script, spath)


class image_callback_T2(keras.callbacks.Callback):
    def __init__(self, X, Y, spath, orig_size, block_size=(25, 25, 25), oversamp=1, lab_trun=4, im_freq=10, batch_size=10):
        self.X = X
        self.Y = Y
        self.spath = spath
        self.orig_size = orig_size
        self.block_size = block_size
        self.oversamp = oversamp
        self.lab_trun = lab_trun
        self.im_freq = im_freq
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):

        if (epoch % self.im_freq == 0):
            print('Starting evaluation')

            # Test the segemntation
            sz = self.X.shape
            test_out = np.zeros_like(self.Y)
            print('\tMaking predictions')
            if len(sz) > 5:
                vols = sz[5]
                for vol in range(vols):
                    temp = self.X[:, :, :, :, :, vol]
                    test_out[:, :, :, :, :, vol] = self.model.predict(x=temp, batch_size=self.batch_size)
            else:

                test_out = self.model.predict(x=self.X, batch_size=self.batch_size)

            # Evaluate metrics
            threshold = model_evaluation(spath=self.spath, Y=self.Y, Y_pred=test_out, epoch=epoch, images=True)
            test_out_thresh = test_out > threshold
            fid = open(os.path.join(self.spath, 'validation_thresholds.txt'), 'w')
            fid.write(str(threshold))
            fid.close()

            # convert volume patches to images
            _, Y = recon_test_3D(X=self.X, Y=self.Y, orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)
            X, test_out_thresh = recon_test_3D(X=self.X, Y=test_out_thresh, orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)

            # Plot images

            Y_mask = np.ma.masked_where(Y.astype(bool) == 0, Y.astype(bool))
            test_out_mask = np.ma.masked_where(test_out_thresh.astype(bool) == 0, test_out_thresh.astype(bool))

            slices = range(20, 40, 4)
            resh = (Y.shape[1] * len(slices), Y.shape[2])
            fig, ax = plt.subplots(nrows=3, ncols=1)

            ax[0].imshow(X[slices, :, :, 0, 0].reshape(resh).T, cmap='gray')
            ax[1].imshow(X[slices, :, :, 0, 0].reshape(resh).T, cmap='gray')
            ax[1].imshow(test_out_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)
            ax[2].imshow(X[slices, :, :, 0, 0].reshape(resh).T, cmap='gray')
            ax[2].imshow(Y_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)

            ax[0].set_title('Input slice (T2). Epoch %d' %epoch)
            ax[1].set_title('Output slice')
            ax[2].set_title('Label')

            ax[2].set_xticklabels(slices)

            for a in ax.ravel():
                a.axis('off')

            fig.savefig(os.path.join(self.spath, 'Val_set_%d.png' %epoch), dpi=300, format='png', frameon=False)

class image_callback_val(keras.callbacks.Callback):
    def __init__(self, X, Y, spath, orig_size, block_size=(25, 25, 25), oversamp=1, lab_trun=4, im_freq=10, batch_size=10):
        self.X = X
        self.Y = Y
        self.spath = spath
        self.orig_size = orig_size
        self.block_size = block_size
        self.oversamp = oversamp
        self.lab_trun = lab_trun
        self.im_freq = im_freq
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):

        if (epoch % self.im_freq == 0) and (epoch != 0):
            print('Starting evaluation')

            # Test the segemntation
            sz = self.X.shape
            test_out = np.zeros_like(self.Y)
            print('\tMaking predictions')
            if len(sz) > 5:
                vols = sz[5]
                for vol in range(vols):
                    temp = self.X[:, :, :, :, :, vol]
                    test_out[:, :, :, :, :, vol] = self.model.predict(x=temp, batch_size=self.batch_size)
            else:

                test_out = self.model.predict(x=self.X, batch_size=self.batch_size)

            # Evaluate metrics
            threshold = model_evaluation(spath=self.spath, Y=self.Y, Y_pred=test_out, epoch=epoch, images=True)
            test_out_thresh = test_out > threshold
            fid = open(os.path.join(self.spath, 'validation_thresholds.txt'), 'w')
            fid.write(str(threshold))
            fid.close()


            # convert volume patches to images
            _, Y = recon_test_3D(X=self.X, Y=self.Y, orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)
            X, test_out_thresh = recon_test_3D(X=self.X, Y=test_out_thresh, orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)

            # Plot images

            Y_mask = np.ma.masked_where(Y.astype(bool) == 0, Y.astype(bool))
            test_out_mask = np.ma.masked_where(test_out_thresh.astype(bool) == 0, test_out_thresh.astype(bool))

            slices = range(20, 40, 4)
            resh = (Y.shape[1] * len(slices), Y.shape[2])
            fig, ax = plt.subplots(nrows=3, ncols=1)

            ax[0].imshow(X[slices, :, :, -1, 0].reshape(resh).T, cmap='gray')
            ax[1].imshow(X[slices, :, :, -1, 0].reshape(resh).T, cmap='gray')
            ax[1].imshow(test_out_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)
            ax[2].imshow(X[slices, :, :, -1, 0].reshape(resh).T, cmap='gray')
            ax[2].imshow(Y_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)

            ax[0].set_title('Input slice (T2). Epoch %d' %epoch)
            ax[1].set_title('Output slice')
            ax[2].set_title('Label')

            ax[2].set_xticklabels(slices)

            for a in ax.ravel():
                a.axis('off')

            fig.savefig(os.path.join(self.spath, 'Val_set_%d.png' %epoch), dpi=300, format='png', frameon=False)

class image_callback_1px(keras.callbacks.Callback):
    def __init__(self, gen, spath, orig_size, block_size=(25, 25, 25), oversamp=1, lab_trun=4, im_freq=10, batch_size=10):
        self.gen = gen
        self.spath = spath
        self.orig_size = orig_size
        self.block_size = block_size
        self.oversamp = oversamp
        self.lab_trun = lab_trun
        self.im_freq = im_freq
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):

        if (epoch % self.im_freq == 0):# and not (epoch == 0):
            print('Starting evaluation')


            # Test the segemntation
            sz = self.X.shape
            test_out = np.zeros_like(self.Y)
            print('\tMaking predictions')
            if len(sz) > 5:
                vols = sz[5]
                for vol in range(vols):
                    temp = self.X[:, :, :, :, :, vol]
                    test_out[:, :, :, :, :, vol] = self.model.predict(x=temp, batch_size=self.batch_size)
            else:

                test_out = self.model.predict(x=self.X, batch_size=self.batch_size)

            # test_out =

            # Evaluate metrics
            threshold = model_evaluation(spath=self.spath, Y=self.Y, Y_pred=test_out, epoch=epoch, images=True)
            test_out_thresh = test_out > threshold
            fid = open(os.path.join(self.spath, 'validation_thresholds.txt'), 'w')
            fid.write(str(threshold))
            fid.close()


            # convert volume patches to images
            # _, Y = recon_test_3D(X=self.X[:int(sz[0]/2 + 1), :, :, :, :], Y=self.Y[:int(sz[0]/2 + 1), :, :, :, :], orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)
            # X, test_out_thresh = recon_test_3D(X=self.X[:int(sz[0]/2 + 1), :, :, :, :], Y=test_out_thresh[:int(sz[0]/2 + 1), :, :, :, :], orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)
            _, Y = recon_test_3D(X=self.X, Y=self.Y, orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)
            X, test_out_thresh = recon_test_3D(X=self.X, Y=test_out_thresh, orig_size=self.orig_size, block_size=self.block_size, oversamp=self.oversamp, lab_trun=self.lab_trun)

            # Plot images

            Y_mask = np.ma.masked_where(Y.astype(bool) == 0, Y.astype(bool))
            test_out_mask = np.ma.masked_where(test_out_thresh.astype(bool) == 0, test_out_thresh.astype(bool))

            slices = range(20, 40, 4)
            resh = (Y.shape[1] * len(slices), Y.shape[2])
            fig, ax = plt.subplots(nrows=3, ncols=1)

            ax[0].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
            ax[1].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
            ax[1].imshow(test_out_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)
            ax[2].imshow(X[slices, :, :, 2, 0].reshape(resh).T, cmap='gray')
            ax[2].imshow(Y_mask[slices, :, :, 0, 0].reshape(resh).T, cmap='summer', alpha=0.5)

            ax[0].set_title('Input slice (T2). Epoch %d' %epoch)
            ax[1].set_title('Output slice')
            ax[2].set_title('Label')

            ax[2].set_xticklabels(slices)

            for a in ax.ravel():
                a.axis('off')

            fig.savefig(os.path.join(self.spath, 'Val_set_%d.png' %epoch), dpi=300, format='png', frameon=False)
