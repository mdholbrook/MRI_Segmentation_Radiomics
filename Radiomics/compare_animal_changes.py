import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()


def load_data(csv_file):
    data = pd.read_csv(csv_file)


    # data = data.dropna(thresh=5)

    # Get file names
    filenames = data['Image']
    inds_1 = [i for (i, name) in enumerate(filenames) if 'T2_1.nii' in name]
    inds_2 = [i for (i, name) in enumerate(filenames) if 'T2_2.nii' in name]

    # Filter out non-feature names
    features = data.keys()
    feature_names = list(
        sorted(filter(lambda k: k.startswith("original_"), features)))
    rad_data = data[feature_names]

    # # Filter out bad features (NaNs or Inf)
    # rad_data = rad_data.dropna(axis=1)
    # infs = (rad_data == np.inf).any()
    # infs = infs[infs == True]
    # rad_data = rad_data.drop(columns=infs.keys())

    return rad_data


def add_rt_to_df(rad_data):
    """
    Append RT labels to radiomic data
    Args:
        rad_data (pandas dataframe): dataframe of radiomic features. Animals are orded by pre/post
            RT so this step is fairly simple.

    Returns:
        (pandas dictionary): radiomic features with appended
    """
    # Get number of samples
    n_samps = rad_data.shape[0]

    # Add a column for Pre/Post RT
    rad_data['RT'] = pd.Series(n_samps//2 * ['Pre'] + n_samps//2 * ['Post'], index=rad_data.index)

    # Add an animal identifier
    rad_data['ID'] = pd.Series(2 * list(range(n_samps//2)))

    return rad_data


def plot_radiomics(rad_data):
    # Make a large plot of values
    rows = np.round(np.sqrt(len(rad_data.keys()))).astype(int)
    cols = np.ceil(len(rad_data.keys())//rows).astype(int)
    fig, ax = plt.subplots(rows, cols, sharex=False)
    fig.set_size_inches(50, 50)
    m = 0

    for (i, ax) in enumerate(ax.ravel()):

        if i < len(rad_data.keys()):
            print(i)
            key = rad_data.keys()[i]

            # rad_data.groupby([key, 'RT']).size().unstack().plot(kind='bar', ax=ax)
            try:
                sns.violinplot('RT', key, data=rad_data, ax=ax, inner='quartile')
                sns.swarmplot(x='RT', y=key, data=rad_data, ax=ax, color='White', edgecolor='gray')

                # rad_data.plot('RT', 'ID', )
                ax.set_ylabel('')
                ax.get_legend().remove()

            except:

                # sns.swarmplot(x='RT', y=key, data=rad_data, ax=ax, color='white', edgecolor='gray')
                # ax.text(0.25, 0.5, 'NaN encountered', fontsize=10)
                jkjkj = 1

            ax.set_title(key)

    fig.savefig('/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/vis.png', dpi=200)


if __name__ == "__main__":
    """
    This script will plot all radiomic features as a function of pre- and post-RT. Useful for viewing all features 
    together.
    """

    csv_file = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/Rad_results_dilated_allCon/radiomic_features_norm.csv'
    rad_data = load_data(csv_file)

    rad_data = add_rt_to_df(rad_data)

    plot_radiomics(rad_data)
