import os
from MachineLearning.load_datasets import load_filenames_2nd, load_data, keep_t2
from glob2 import glob
import nibabel as nib
import numpy as np
import keras
from Segmentation.model_keras import *
from sklearn.metrics import precision_recall_curve, precision_score, \
    recall_score, roc_auc_score, f1_score, \
    precision_recall_fscore_support, matthews_corrcoef, jaccard_similarity_score, accuracy_score
import pandas as pd
from pylab import rcParams
import seaborn as sns


# Set up plotting properties
sns.set(style='ticks', palette='Spectral', font_scale=1.5)
rcParams['figure.figsize'] = 6, 4

RAND_SEED = 42


def load_test_volumes(only_t2=False, adaptive_hist=False):

    # Set up image path
    image_base_path = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma'

    # Set up data constants
    block_size = [18, 142, 142]
    oversamp_test = 1.0
    lab_trun = 2
    test_split = 0.1

    # Get filenames
    filenames = load_filenames_2nd(base_path=image_base_path)
    nfiles = len(filenames)

    if only_t2:
        filenames = keep_t2(filenames)

    # Remove validation and test set
    inds = np.array((range(nfiles)), dtype=int)
    np.random.seed(RAND_SEED)
    np.random.shuffle(inds)

    # Test data
    test_inds = inds[-round(test_split*nfiles):]
    test_files = [filenames[i] for i in test_inds]

    # Yield the number of sets in the generator
    yield test_files

    for test_file in test_files:

        X_test, Y_test, orig_size_test = load_data([test_file], block_size,
                                                   oversamp_test,
                                                   lab_trun, adaptive_hist)

        yield [X_test, Y_test, orig_size_test]


def load_models(paths):
    """
    Loads a list of models
    Args:
        paths (list): list of paths to models (not including the filename)

    Returns:

    """

    model = []

    for path in paths:

        model_path = os.path.join(path, 'Trained_model.h5')
        model.append(keras.models.load_model(model_path,
                                             custom_objects={'dice_loss': dice_loss,
                                                             'dice_metric': dice_metric}))

    return model


def score_pred(Y_lab, Y_prob, threshold):
    """
    Calculate a set of scores for the predictions
    Args:
        Y_lab (numpy array): labels
        Y_prob (numpy array): predictions as probablilities
        threshold (float): threshold for predictions

    Returns:
        (float): precision
        (float): recall
        (float): f1 score (Dice)
        (float): support
        (float): volume overlap error
        (float): binary accuracy
    """
    Y_thresh = Y_prob >= threshold

    precision = []
    recall = []
    fbeta_score = []
    support = []
    voe = []
    acc = []

    Y_lab = Y_lab.reshape(-1)
    Y_thresh = Y_thresh.reshape(-1)

    # Compute precision/recall scores
    scores = precision_recall_fscore_support(y_true=Y_lab, y_pred=Y_thresh)
    precision.append(scores[0][1])
    recall.append(scores[1][1])
    fbeta_score.append(scores[2][1])
    support.append(scores[3][1]/(scores[3][0] + scores[3][1]))  # percent of volume occupied by tumor
    voe.append(jaccard_similarity_score(y_true=Y_lab, y_pred=Y_thresh))
    acc.append(accuracy_score(y_true=Y_lab, y_pred=Y_thresh, normalize=True))

    return precision, recall, fbeta_score, support, voe, acc


def plot_results_cat(df, spath):

    # Remove support and accuracy
    df = df.loc[df['Metric'] !='Accuracy']
    df = df.loc[df['Metric'] !='Support']

    # Plot results
    plt.figure(1)
    sns.swarmplot(x='Metric', y='Value', hue='Data', data=df, size=10, dodge=True)
    # plt.grid()
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'cat_plot_1901.svg'), dpi=300)

    plt.show()


def plot_results(df, spath):

    plt.figure(11)
    sns.swarmplot(x='Data', y='Precision', data=df, size=10)
    plt.grid()
    plt.ylabel('Precision Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'prec_1901.svg'), dpi=300)

    plt.figure(12)
    sns.swarmplot(x='Data', y='Recall', data=df, size=10)
    plt.grid()
    plt.ylabel('Recall Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'rec_1901.svg'), dpi=300)

    plt.figure(13)
    sns.swarmplot(x='Data', y='dice', data=df, size=10)
    plt.grid()
    plt.ylabel('DICE Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'dice_1901.svg'), dpi=300)

    plt.figure(14)
    sns.swarmplot(x='Data', y='Support', data=df, size=10)
    plt.grid()
    plt.ylabel('Support Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'support_1901.svg'), dpi=300)

    plt.figure(15)
    sns.swarmplot(x='Data', y='VOE', data=df, size=10)
    plt.grid()
    plt.ylabel('VOE Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'VOE_1901.svg'), dpi=300)

    plt.figure(16)
    sns.swarmplot(x='Data', y='Accuracy', data=df, size=10)
    plt.grid()
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(spath, 'accuracy_1901.svg'), dpi=300)

    plt.show()


def write_stats(df, thresholds, spath):

    f = open(os.path.join(spath, 'stats.txt'), 'w')
    f.write('T2\n')
    f.write(20*'-' + '\n')
    f.write('Threshold: %0.3f\n' % thresholds[0])
    f.write('Mean:\n')
    a = df.loc[df['Data'] == 'T2']
    f.write(a.mean().to_string())
    f.write('\n\nStd:\n')
    f.write(a.std().to_string())
    f.write(3*'\n')

    a = df.loc[df['Data'] == 'T1, T1C, T2']
    f.write('T1, T1C, T2\n')
    f.write(20*'-' + '\n')
    f.write('Threshold: %0.3f\n' % thresholds[1])
    f.write('Mean:\n')
    f.write(a.mean().to_string())
    f.write('\n\nStd:\n')
    f.write(a.std().to_string())

    f.close()


def clear_vol_stats():

    f = open(os.path.join(spath, 'volumes.txt'), 'w')
    f.close()


def write_volumes(Y, Y_pred, spath):

    y_vol = Y.sum()
    pred_vol = Y_pred.sum()

    f = open(os.path.join(spath, 'volumes.txt'), 'a')
    f.write('Label: %d\tPrediction: %d\tPercent:\%0.3f\n' % (y_vol, pred_vol, 100*y_vol/pred_vol))
    f.close()


def save_df(df, spath, descriptor):

    df.to_csv(os.path.join(spath, 'data_%s.csv' % descriptor))


def run_from_df(spath):

    # Load concatenated dataframe
    template = '*cat.csv'
    file = glob(os.path.join(spath, template))

    df_cat = pd.DataFrame.from_csv(file)

    # Load metrics dataframe
    template = '*metrics.csv'
    file = glob(os.path.join(spath, template))

    df = pd.DataFrame.from_csv(file)

    # Write statistics
    write_stats(df, spath)

    # Plot results
    plot_results(df, spath)
    plot_results_cat(df_cat, spath)


def main(paths, spath):
    """

    Args:
        paths (list of str): path to t2_only and all_contrast models
        thresholds (list of float): training thresholds

    Returns:

    """

    # Set up data constants
    block_size = [18, 142, 142]
    oversamp_test = 1.0
    lab_trun = 2
    test_split = 0.1

    # Load models
    models = load_models(paths)

    # Set up data generator
    gen_t2 = load_test_volumes(only_t2=True)
    gen = load_test_volumes()
    nsets = next(gen)
    nsets = next(gen_t2)

    print('Testing using {} sets'.format(nsets))

    # Set up metric lists
    df = pd.DataFrame(columns=['Loss', 'Data', 'Precision', 'Recall', 'Dice', 'Support', 'VOE', 'Accuracy'])
    df_cat = pd.DataFrame(columns=['Loss', 'Data', 'Metric', 'Value'])  # Concatenated dataframe
    contrasts = ['Multi-modal', 'T2 only']
    con_lab = ['t2', 'all']

    # Process
    clear_vol_stats()
    thresholds = []
    flag = True
    z = 0
    while flag:

        try:
            print('\tVolume %d' % (z + 1))
            print('Loading test batch')
            [xall, yall, szall] = next(gen)
            [xt2, yt2, szt2] = next(gen_t2)

            for model, path in zip(models, paths):

                # Load model threshold
                file = os.path.join(path, 'metrics2.txt')
                with open(file, 'r') as f:
                    dat = f.readlines()

                thr_ind = -7
                tmp = [i for i in dat[thr_ind] if i.isdigit() or i == '.']
                threshold = float(''.join(tmp))
                thresholds.append(threshold)

                # Get model loss
                if 'dice' in path.lower():
                    loss = 'Dice'
                else:
                    loss = 'Xentropy'

                # Get skip status
                if 'skip' in path.lower():
                    skip = 'Yes'
                else:
                    skip = 'No'

                # Get number of model inputs
                mod_input_ch = model.input_shape[-1]

                # Get correct contrast
                if mod_input_ch == 1:
                    x, y, sz = xt2, yt2, szt2
                    contrast = contrasts[1]
                else:
                    x, y, sz = xall, yall, szall
                    contrast = contrasts[0]


                # Predict using model
                print('Making predictions')
                y_pred = model.predict(x)

                # Compute metrics
                print('Evaluating predictions')
                res = score_pred(y, y_pred, threshold)

                # Concatenate metrics
                df = df.append(pd.DataFrame({'Loss': loss,
                                             'Data': contrast,
                                             'Skip': skip,
                                             'Precision': res[0],
                                             'Recall': res[1],
                                             'Dice': res[2],
                                             'Support': res[3],
                                             'VOE': res[4],
                                             'Accuracy': res[5]
                                             }))

                for ii in range(6):
                    df_cat = df_cat.append(pd.DataFrame({'Loss': loss,
                                                         'Data': contrast,
                                                         'Metric': df.keys()[ii+3],
                                                         'Value': res[ii]
                                                         }))

                # Reconstruct images
                # _, y = recon_test_3D(X=x, Y=y, orig_size=sz, block_size=block_size, oversamp=oversamp_test,
                #                      lab_trun=lab_trun)
                # x, y_pred = recon_test_3D(X=x, Y=y_pred, orig_size=sz, block_size=block_size, oversamp=oversamp_test,
                #                           lab_trun=lab_trun)
                #
                # # Swap axes
                # x = np.rollaxis(x, 0, 2).swapaxes(1, 2)
                # y = np.rollaxis(y, 0, 2).swapaxes(1, 2)
                # y_pred = np.rollaxis(y_pred, 0, 2).swapaxes(1, 2)

                # Threshold segmentation
                y_thresh = y_pred > threshold

                # Record volume measurements
                write_volumes(y, y_thresh, spath)


            z += 1

        except StopIteration:

            print('Exhausted generator')
            flag = False

    # Plot results
    # print('Saving plots')
    # plot_results_cat(df_cat, spath)
    # plot_results(df, spath)

    # Write statistics
    write_stats(df, thresholds, spath)

    # Update dataframes to include stds
    losses = df['Loss'].unique().tolist()
    datas = df['Data'].unique().tolist()
    skips = df['Skip'].unique().tolist()
    metrics = ['Accuracy', 'Dice', 'Precision', 'Recall', 'Support', 'VOE']
    df_out = {i: [] for i in df.keys()}
    for loss in losses:
        ind1 = df['Loss'] == loss
        for data in datas:
            ind2 = df['Data'] == data
            for skip in skips:
                ind3 = df['Skip'] == skip

                # Create output df
                df_out['Loss'].append(loss)
                df_out['Data'].append(data)
                df_out['Skip'].append(skip)

                for metric in metrics:
                    # Get measurements
                    vals = df.loc[ind1 & ind2 & ind3, metric]

                    df_out[metric].append('{:0.3f} \xb1 {:0.3f}'.format(vals.mean(), vals.std()))
                    # std_metric = metric + '_std'
                    # df_out[std_metric].append(vals.std())

    df_out = pd.DataFrame.from_dict(df_out)


    # Save dataframes
    print('Saving data')
    save_df(df_out, spath, descriptor='metrics')
    save_df(df_cat, spath, descriptor='cat')


if __name__ == '__main__':
    """
    Example of how to test train networks.
    """

    paths = ['/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_08_14-36-46_cnn_model_3D_3lyr_relu_dice',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_08_21-50-21_cnn_model_3D_3lyr_do_relu_dice_skip',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_06-49-45_cnn_model_3D_3lyr_do_relu_xentropy',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_14-12-47_cnn_model_3D_3lyr_do_relu_xentropy_skip',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_09_23-04-28_t2_cnn_model_3D_3lyr_relu_dice',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_04-50-05_t2_cnn_model_3D_3lyr_do_relu_dice_skip',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_12-28-23_t2_cnn_model_3D_3lyr_do_relu_xentropy',
             '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/2019_11_10_18-43-24_t2_cnn_model_3D_3lyr_do_relu_xentropy_skip']

    spath = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/Analysis/Segmentation_images'
    main(paths, spath)
