import sys, os
import keras.backend.tensorflow_backend as ktf
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import datetime
from scipy import interp
from datetime import datetime
import datetime as dt
from Segmentation.model_keras import *
from Segmentation.load_datasets import load_filenames_2nd, load_data, keep_t2
from sklearn.metrics import auc, roc_curve
from glob2 import glob
import nibabel as nib
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")


RAND_SEED = 42


# Disable stdout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore stdout
def enablePrint():
    sys.stdout = sys.__stdout__


def train_model(networks, spaths, only_t2):
    """
    Train specified model
    Args:
        networks (list): list of keras networks to train
        spaths (list): list of output directories
        only_t2 (bool): whether or not to only use T2 data

    Returns:

    """

    epochs = 600
    batch_size = 20
    block_size = [18, 142, 142]
    oversamp = 1.0
    oversamp_test = 1.0
    lab_trun = 2
    im_freq = 50
    val_split = 0.2
    test_split = 0.1
    lr = 1e-4

    adaptive_hist = False

    # Load training data
    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'
    # image_base_path = 'E:/MR Data/MR_Images_Sarcoma'

    # Load training data
    filenames = load_filenames_2nd(base_path=image_base_path)
    nfiles = len(filenames)

    # Remove T2 images
    if only_t2:
        filenames = keep_t2(filenames)

    # Remove validation and test set
    inds = np.array((range(nfiles)), dtype=int)
    np.random.seed(RAND_SEED)
    np.random.shuffle(inds)

    # Validation data
    val_inds = inds[:round(val_split*nfiles)]
    val_file = [filenames[i] for i in val_inds]

    # Test data
    test_inds = inds[-round(test_split*nfiles):]
    test_file = [filenames[i] for i in test_inds]

    # Delete all data
    filenames = [filename for i, filename in enumerate(filenames) if i not in
                 list(val_inds) + list(test_inds)]


    # Load data
    print('Loading data')
    x, y, orig_size = load_data(filenames, block_size, oversamp,
                                lab_trun, adaptive_hist)

    # val_file = val_file[:1]
    x_val, Y_val, orig_size_val = load_data(val_file, block_size, oversamp,
                                            lab_trun, adaptive_hist)

    # test_file = test_file[:1]
    x_test, y_test, orig_size_test = load_data(test_file, block_size,
                                               oversamp_test,
                                               lab_trun, adaptive_hist)

    # shuffle training data
    inds = np.arange(0, x.shape[0])
    np.random.seed(5)
    np.random.shuffle(inds)
    x = x[inds]
    y = y[inds]

    print('Size of training set:\t\t', x.shape)
    print('Size of validation set: \t', x_val.shape)
    print('Size of test set: \t\t', x_test.shape)

    sz_patch = x.shape

    for network, spath in zip(networks, spaths):

        # Set up save path
        spath = spath % datetime.strftime(datetime.now(), '%Y_%m_%d_%H-%M-%S')

        if not os.path.exists(spath):
            os.mkdir(spath)

        # Display which network is training
        _, net_name = os.path.split(spath)
        if only_t2:
            net_name += '_t2'
        print('\n\n\n')
        print('Training: %s' % net_name)
        print('-' * 80 + '\n')

        # Save a copy of this code
        save_code(spath, os.path.realpath(__file__))

        # Load model
        model, opt = network(pretrained_weights=None,
                             input_size=(sz_patch[1],
                                         sz_patch[2],
                                         sz_patch[3],
                                         sz_patch[4]),
                             lr=lr)

        # Set up callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir="logs/%s_%s"
                                                  % net_name,
                                                  write_graph=False,
                                                  write_grads=False,
                                                  write_images=False,
                                                  histogram_freq=0)
        ckpoint_weights = keras.callbacks.ModelCheckpoint(os.path.join(spath, 'ModelCheckpoint.h5'),
                                                          monitor='val_loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          mode='auto',
                                                          period=10)

        image_recon_callback = image_callback_val(x_val, Y_val, spath,
                                                  orig_size_val,
                                                  block_size=block_size,
                                                  oversamp=oversamp_test,
                                                  lab_trun=lab_trun,
                                                  im_freq=im_freq,
                                                  batch_size=batch_size)


        # Train model
        model.fit(x=x, y=y,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(x_val, Y_val),
                  callbacks=[tensorboard,
                             ckpoint_weights,
                             image_recon_callback
                             ]
                  )

        model.save(os.path.join(spath, 'Trained_model.h5'))

        # Calculate best threshold from training data
        threshold = training_threshold(model, spath, X=x, Y=y)
        # threshold = 0.5

        # Evaluate test data
        test_set_3D(model, x_test, y_test, spath, orig_size_test, block_size,
                    oversamp_test, lab_trun, batch_size=1, threshold=threshold)


def run_model_test(spaths, only_t2):
    """
    Test trained network
    Args:
        spaths (list): list of paths which contain the trained models
        only_t2 (bool): whether or not to only use T2 data

    Returns:

    """

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

    adaptive_hist = False

    # Load training data
    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'

    # Load training data
    filenames = load_filenames_2nd(base_path=image_base_path)
    nfiles = len(filenames)

    # Remove all but T2 images
    if only_t2:
        filenames = keep_t2(filenames)

    # Remove validation and test set
    inds = np.array((range(nfiles)), dtype=int)
    np.random.seed(RAND_SEED)
    np.random.shuffle(inds)

    # Validation data
    val_inds = inds[:round(val_split*nfiles)]
    val_file = [filenames[i] for i in val_inds]

    # Test data
    test_inds = inds[-round(test_split*nfiles):]
    test_file = [filenames[i] for i in test_inds]

    # Delete all data
    filenames = [filename for i, filename in enumerate(filenames) if i not in
                 list(val_inds) + list(test_inds)]


    # Load data
    x, y, orig_size = load_data(filenames, block_size, oversamp,
                                lab_trun, adaptive_hist)

    x_test, y_test, orig_size_test = load_data(test_file, block_size,
                                               oversamp_test,
                                               lab_trun, adaptive_hist)
    print('Size of training set:\t\t', x.shape)
    print('Size of test set: \t\t', x_test.shape)

    for spath in spaths:

        # Display which network is training
        _, net_name = os.path.split(spath)
        print('\n\n\n')
        print('Testing: %s' % net_name)
        print('-' * 80 + '\n')

        # Load trained model
        model_path = os.path.join(spath, 'Trained_model.h5')
        model = keras.models.load_model(model_path,
                                        custom_objects={'dice_loss': dice_loss,
                                                        'dice_metric': dice_metric})

        # Calculate best threshold from training data
        threshold = training_threshold(model, spath, X=x, Y=y)
        # threshold = 0.5

        # Evaluate test data
        test_set_3D(model, x_test, y_test, spath, orig_size_test, block_size,
                    oversamp_test, lab_trun, batch_size=20, threshold=threshold, vols=len(test_file), continuous=True)


def run_model_test_best_weigts(spaths, only_t2):

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

    adaptive_hist = False

    # Load training data
    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'

    # Load training data
    filenames = load_filenames_2nd(base_path=image_base_path)
    nfiles = len(filenames)

    # Remove all but T2 images
    if only_t2:
        filenames = keep_t2(filenames)

    # Remove validation and test set
    inds = np.array((range(nfiles)), dtype=int)
    np.random.seed(RAND_SEED)
    np.random.shuffle(inds)

    # Validation data
    val_inds = inds[:round(val_split*nfiles)]
    val_file = [filenames[i] for i in val_inds]

    # Test data
    test_inds = inds[-round(test_split*nfiles):]
    test_file = [filenames[i] for i in test_inds]

    # Delete all data
    filenames = [filename for i, filename in enumerate(filenames) if i not in
                 list(val_inds) + list(test_inds)]


    # Load data
    x, y, orig_size = load_data(filenames, block_size, oversamp,
                                lab_trun, adaptive_hist)

    x_test, y_test, orig_size_test = load_data(test_file, block_size,
                                               oversamp_test,
                                               lab_trun, adaptive_hist)
    print('Size of training set:\t\t', x.shape)
    print('Size of test set: \t\t', x_test.shape)

    for spath in spaths:

        # Display which network is training
        _, net_name = os.path.split(spath)
        print('\n\n\n')
        print('Testing: %s' % net_name)
        print('-' * 80 + '\n')

        # Load trained model
        model_path = os.path.join(spath, 'Trained_model.h5')
        model = keras.models.load_model(model_path,
                                        custom_objects={'dice_loss': dice_loss,
                                                        'dice_metric': dice_metric})

        # Load best (from validation set)
        weights_file = os.path.join(spath, 'ModelCheckpoint.h5')
        model.load_weights(weights_file)

        # Calculate best threshold from training data
        threshold = training_threshold(model, spath, X=x, Y=y)
        # threshold = 0.5

        # Evaluate test data
        test_set_3D(model, x_test, y_test, spath, orig_size_test, block_size,
                    oversamp_test, lab_trun, batch_size=20, threshold=threshold, vols=len(test_file), continuous=True)


def train_networks():
    """
    Sets up all networks to train
    Returns:

    """

    # Save locations
    save_base_path = '/media/matt/Seagate Expansion Drive/MR Data/ML_Results'

    # List of networks to train
    networks = [cnn_model_3D_3lyr_relu_dice,
                cnn_model_3D_3lyr_do_relu_dice_skip,
                #cnn_model_3D_FlDense_dice,
                #cnn_model_3D_3lyr_do_relu_hing,
                cnn_model_3D_3lyr_do_relu_xentropy,
                cnn_model_3D_3lyr_do_relu_xentropy_skip
                ]

    spaths = ['%s_' + i.__name__ for i in networks]
    spaths = [os.path.join(save_base_path, path) for path in spaths]

    # Train the model
    train_model(networks, spaths, only_t2=False)

    # T2 - only
    # List of networks to train
    networks = [cnn_model_3D_3lyr_relu_dice,
                cnn_model_3D_3lyr_do_relu_dice_skip,
                # cnn_model_3D_FlDense_dice,
                # cnn_model_3D_3lyr_do_relu_hing,
                cnn_model_3D_3lyr_do_relu_xentropy,
                cnn_model_3D_3lyr_do_relu_xentropy_skip
                ]

    spaths = ['%s_t2_' + i.__name__ for i in networks]
    spaths = [os.path.join(save_base_path, path) for path in spaths]

    # Train the model
    train_model(networks, spaths, only_t2=True)


def run_networks_test():
    """
    Tests all trained networks.
    Returns:

    """
    # Paths to trained networks
    spaths = ['/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_08_14-36-46_cnn_model_3D_3lyr_relu_dice',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_08_21-50-21_cnn_model_3D_3lyr_do_relu_dice_skip',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_06-49-45_cnn_model_3D_3lyr_do_relu_xentropy',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_14-12-47_cnn_model_3D_3lyr_do_relu_xentropy_skip',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_23-04-28_t2_cnn_model_3D_3lyr_relu_dice',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_04-50-05_t2_cnn_model_3D_3lyr_do_relu_dice_skip',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_12-28-23_t2_cnn_model_3D_3lyr_do_relu_xentropy',
              '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_18-43-24_t2_cnn_model_3D_3lyr_do_relu_xentropy_skip'
              ]

    # run_model_test(spaths[:4], only_t2=False)
    # run_model_test(spaths[4:], only_t2=True)

    run_model_test_best_weigts(spaths[:4], only_t2=False)
    run_model_test_best_weigts(spaths[4:], only_t2=True)


def load_local_test_data(filenames):

    X = np.empty(shape=(0, 0, 0, 0, 0))
    for file in filenames:

        tmpx = nib.load(file).get_data().astype(np.float).squeeze()

        try:
            X = np.concatenate((X, tmpx), axis=2)
        except ValueError:
            X = tmpx

    return X


def load_train_volumes(only_t2=False, adaptive_hist=False):

    # Set up image path
    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'

    # Set up data constants
    block_size = [18, 142, 142]
    oversamp_test = 1.0
    lab_trun = 2
    test_split = 0.1
    val_split = 0.2

    # Get filenames
    filenames = load_filenames_2nd(base_path=image_base_path)
    nfiles = len(filenames)

    # Yield the number of sets in the generator
    yield round((1 - (val_split + test_split) ) * nfiles)

    if only_t2:
        filenames = keep_t2(filenames)

    # Remove validation and test set
    inds = np.array((range(nfiles)), dtype=int)
    np.random.seed(RAND_SEED)
    np.random.shuffle(inds)
    mask = np.ones(inds.shape, dtype=bool)

    # Test data
    mask[:round(val_split * nfiles)] = 0
    mask[-round(test_split*nfiles):] = 0
    train_files = [filenames[i] for i in train_inds]

    while True:

        for train_file in train_files:

            X_train, Y_train, _ = load_data([train_file], block_size,
                                                       oversamp_test,
                                                       lab_trun, adaptive_hist)

            yield [X_train, Y_train]


def load_models(paths):
    """
    Loads a list of models
    Args:
        paths (list): list of paths to models (not including the filename)

    Returns:

    """

    model = []

    for path in paths:

        model_name = os.path.join(path, 'Trained_model.h5')

        model.append(keras.models.load_model(model_name,
                                             custom_objects=
                                             {'dice_loss': dice_loss}))

    return model


def read_tensorboard(path, fields, label):
    """
    Reads tensorboard files and returns data requests as a pandas array
    Args:
        path (str): path to tensorboard file
        fields (list of str): fields to return
        label (str): dataset identifier

    Returns:
        pandas dataframe
    """

    # Update fields with validation data
    fields = [[i] + ['val_{}'.format(i)] for i in fields]
    fields = sum(fields, [])

    # Set up output dictionary
    df = dict()
    for field in fields:
        df[field] = list()

    # Read the file
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            for field in fields:
                if field == v.tag:
                    df[field].append(v.simple_value)

    # Convert to pandas
    df = pd.DataFrame.from_dict(df)

    # Append set label
    df['label'] = label
    df['Epoch'] = list(range(1, len(df)+1))

    return df


def training_curves(spath):
    """
    Load and plot training curves.
    Args:
        spath (str): path on which to save output

    Returns:

    """

    # Location with saved curves
    dat_path = ['/home/matt/Documents/SegSarcoma/logs/2019_11_08_14-36-47_2019_11_08_14-36-46_cnn_model_3D_3lyr_relu_dice',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_08_21-50-22_2019_11_08_21-50-21_cnn_model_3D_3lyr_do_relu_dice_skip',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_09_06-49-45_2019_11_09_06-49-45_cnn_model_3D_3lyr_do_relu_xentropy',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_09_14-12-48_2019_11_09_14-12-47_cnn_model_3D_3lyr_do_relu_xentropy_skip',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_09_23-04-28_2019_11_09_23-04-28_t2_cnn_model_3D_3lyr_relu_dice_t2',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_10_04-50-05_2019_11_10_04-50-05_t2_cnn_model_3D_3lyr_do_relu_dice_skip_t2',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_10_12-28-24_2019_11_10_12-28-23_t2_cnn_model_3D_3lyr_do_relu_xentropy_t2',
                '/home/matt/Documents/SegSarcoma/logs/2019_11_10_18-43-24_2019_11_10_18-43-24_t2_cnn_model_3D_3lyr_do_relu_xentropy_skip_t2'
                ]
    labels = ['DICE', 'DICE - skip', 'XEntr', 'XEntr - skip', 'Dice T2', 'Dice T2 - skip', 'XEntr T2', 'XEntr T2 - skip']
    fields = ['binary_accuracy', 'dice_metric', 'loss']

    # Initialize dataframe
    df = pd.DataFrame()

    # Load data
    for dpath, label in zip(dat_path, labels):

        # Get tensorboard file
        file = glob(os.path.join(dpath, 'events*'))[0]

        # Read data
        tmp = read_tensorboard(file, fields, label)

        # Append data
        df = pd.concat((df, tmp))

    # Save path
    path = '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/JournalPaper/TrainingCurves'

    # Plot with and without skip connections
    sns.set('paper')
    sns.set_style('whitegrid')
    fs = 12
    lw = 2.5

    # Labels
    labs = ['DICE', 'DICE Skip', 'XEntropy', 'XEntropy - Skip']
    colors = ['b', 'r', 'g', 'k']

    snames = os.path.join(path, 'Acc.svg')

    plt.close()
    fig, ax = plt.subplots()
    for z in range(len(labels) // 2):
        inds = df['label'] == labels[z]
        ax.plot(df.loc[inds]['binary_accuracy'], '-', color=colors[z], lw=lw, label=labels[z])
        ax.plot(df.loc[inds]['val_binary_accuracy'], ':', color=colors[z], lw=lw, label='Validation')

        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('Accuracy', fontsize=fs)
        ax.set_ylim([0.9, 1.0])
        # ax.set_yticks(np.linspace(0.9, 1.0, 3))
        ax.legend(fontsize=fs-1)
        for item in list(ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs-2)
    fig.savefig(os.path.join(path, snames))

    # DICE
    snames = os.path.join(path, 'Dice_metric.svg')

    fig, ax = plt.subplots()
    for z in range(len(labels) // 2):
        inds = df['label'] == labels[z]
        ax.plot(df.loc[inds]['dice_metric'], '-', color=colors[z], lw=lw, label=labels[z])
        ax.plot(df.loc[inds]['val_dice_metric'], ':', color=colors[z], lw=lw, label='Validation')
        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('DICE', fontsize=fs)
        ax.set_ylim([0.35, 1.0])
        ax.legend(fontsize=fs-1, loc=4)
        for item in list(ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs-2)
    fig.savefig(os.path.join(path, snames))

    # Loss
    snames = os.path.join(path, 'Loss.svg')

    fig, ax = plt.subplots()
    for z in range(len(labels) // 2):
        inds = df['label'] == labels[z]
        ax.plot(df.loc[inds]['loss'], '-', color=colors[z], lw=lw, label=labels[z])
        ax.plot(df.loc[inds]['val_loss'], ':', color=colors[z], lw=lw, label='Validation')
        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('Loss', fontsize=fs)
        ax.set_ylim([0.0, 0.65])
        ax.legend(fontsize=fs-1, loc=1)
        for item in list(ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs-2)
    fig.savefig(os.path.join(path, snames))

    ## T2 ##
    snames = os.path.join(path, 'Acc_t2.svg')

    plt.close()
    fig, ax = plt.subplots()
    for z in range(len(labels) // 2, len(labels)):
        inds = df['label'] == labels[z]
        ax.plot(df.loc[inds]['binary_accuracy'], '-', color=colors[z-len(labels)//2], lw=lw, label=labels[z])
        ax.plot(df.loc[inds]['val_binary_accuracy'], ':', color=colors[z-len(labels)//2], lw=lw, label='Validation')

        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('Accuracy', fontsize=fs)
        ax.set_ylim([0.9, 1.0])
        # ax.set_yticks(np.linspace(0.9, 1.0, 3))
        ax.legend(fontsize=fs - 1)
        for item in list(ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs - 2)
    fig.savefig(os.path.join(path, snames))

    # DICE
    snames = os.path.join(path, 'Dice_metric_t2.svg')

    fig, ax = plt.subplots()
    for z in range(len(labels) // 2, len(labels)):
        inds = df['label'] == labels[z]
        ax.plot(df.loc[inds]['dice_metric'], '-', color=colors[z-len(labels)//2], lw=lw, label=labels[z])
        ax.plot(df.loc[inds]['val_dice_metric'], ':', color=colors[z-len(labels)//2], lw=lw, label='Validation')
        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('DICE', fontsize=fs)
        ax.set_ylim([0.35, 1.0])
        ax.legend(fontsize=fs - 1, loc=4)
        for item in list(ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs - 2)
    fig.savefig(os.path.join(path, snames))

    # Loss
    snames = os.path.join(path, 'Loss_t2.svg')

    fig, ax = plt.subplots()
    for z in range(len(labels) // 2, len(labels)):
        inds = df['label'] == labels[z]
        ax.plot(df.loc[inds]['loss'], '-', color=colors[z-len(labels)//2], lw=lw, label=labels[z])
        ax.plot(df.loc[inds]['val_loss'], ':', color=colors[z-len(labels)//2], lw=lw, label='Validation')
        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('Loss', fontsize=fs)
        ax.set_ylim([0.0, 0.65])
        ax.legend(fontsize=fs - 1, loc=1)
        for item in list(ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs - 2)
    fig.savefig(os.path.join(path, snames))


def compare_dice():
    """
    Load and make plots of Dice score for each network trained.
    Returns:

    """

    folders = ['/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_08_14-36-46_cnn_model_3D_3lyr_relu_dice',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_08_21-50-21_cnn_model_3D_3lyr_do_relu_dice_skip',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_06-49-45_cnn_model_3D_3lyr_do_relu_xentropy',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_14-12-47_cnn_model_3D_3lyr_do_relu_xentropy_skip',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_23-04-28_t2_cnn_model_3D_3lyr_relu_dice',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_04-50-05_t2_cnn_model_3D_3lyr_do_relu_dice_skip',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_12-28-23_t2_cnn_model_3D_3lyr_do_relu_xentropy',
               '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_18-43-24_t2_cnn_model_3D_3lyr_do_relu_xentropy_skip']

    labels = ['DICE', 'DICE - skip', 'XEntropy', 'XEntropy - skip', 'Dice T2', 'Dice T2 - skip', 'XEntropy T2', 'Xentropy T2 - skip']


    file_name = 'metrics2.txt'

    df = {'Net':[], 'Threshold': [], 'Precision': [], 'Recall': [], 'ROC': [], 'DICE': [], 'VOE': []}

    keys = ['Threshold', 'ROC', 'Precision', 'Recall', 'DICE', 'VOE']

    for folder, label in zip(folders, labels):

        file = os.path.join(folder, file_name)

        df['Net'].append(label)

        # Read meetrics file
        with open(file, 'r') as f:
            dat = f.readlines()

        # Append values
        for ind, k in zip(range(-7, -1), keys):

            tmp = [i for i in dat[ind] if i.isdigit() or i == '.']
            df[k].append(float(''.join(tmp)))


    # Compute VOE
    from sklearn.metrics import jaccard_similarity_score

    # Convert to Pandas
    df = pd.DataFrame.from_dict(df)
    df = df.reindex([0, 1, 4, 5, 2, 3, 6, 7])

    # Write to CSV
    df.to_csv('/media/matt/Seagate Expansion Drive/MR Data/ML_Results/JournalPaper/metrics-val.csv')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.2, 0.80, 0.47])


    g = sns.barplot(x='Net', y='DICE', data=df, ax=ax, palette='Set1')
    plt.xticks(rotation=45)
    fig.savefig('/media/matt/Seagate Expansion Drive/MR Data/ML_Results/JournalPaper/DICE_nets.svg')

    plt.close(fig)


def statistical_metrics(spaths):

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

    # Load training data
    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'

    # Load training data
    filenames = load_filenames_2nd(base_path=image_base_path)
    nfiles = len(filenames)

    # Remove all but T2 images
    if only_t2:
        filenames = keep_t2(filenames)

    # Remove validation and test set
    inds = np.array((range(nfiles)), dtype=int)
    np.random.seed(RAND_SEED)
    np.random.shuffle(inds)

    # Validation data
    val_inds = inds[:round(val_split*nfiles)]
    val_file = [filenames[i] for i in val_inds]

    # Test data
    test_inds = inds[-round(test_split*nfiles):]
    test_file = [filenames[i] for i in test_inds]

    # Delete all data
    filenames = [filename for i, filename in enumerate(filenames) if i not in
                 list(val_inds) + list(test_inds)]


    # Load data
    x_test, y_test, orig_size_test = load_data(test_file, block_size,
                                               oversamp_test,
                                               lab_trun, adaptive_hist)
    print('Size of test set: \t\t', x_test.shape)

    for spath in spaths:

        # Display which network is training
        _, net_name = os.path.split(spath)
        print('\n\n\n')
        print('Testing: %s' % net_name)
        print('-' * 80 + '\n')

        # Load trained model
        model_path = os.path.join(spath, 'Trained_model.h5')
        model = keras.models.load_model(model_path,
                                        custom_objects={'dice_loss': dice_loss,
                                                        'dice_metric': dice_metric})

        # Load best threshold
        file = os.path.join(spath, 'metrics2.txt')

        # Read meetrics file
        with open(file, 'r') as f:
            dat = f.readlines()

        # Append values
        ind = -7
        tmp = [i for i in dat[ind] if i.isdigit() or i == '.']
        threshold = float(''.join(tmp))

        # Evaluate test data



if __name__ == '__main__':
    tstart = time.time()
    train_networks()
    run_networks_test()
    training_curves(spath='/media/matt/Seagate Expansion Drive/MR Data/ML_Results/JournalPaper/TrainingCurves')
    compare_dice()

    print('\tTotal time (HH:MM:SS): %s\n\n' % (str(dt.timedelta(seconds=round(time.time() - tstart)))))
