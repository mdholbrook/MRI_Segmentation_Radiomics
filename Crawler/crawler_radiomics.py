import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from skimage.morphology import ball, binary_dilation, binary_closing
from datetime import datetime
import logging
import os
import sys
import pandas
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from time import time


def gen_savename(file, save_base_path, save_type, ext='.nii'):

    sfiles = os.path.join(save_base_path, save_type + ext + '.gz')
    ids = 'working'

    return sfiles, ids


def gen_images_csv(im_files, mask_file, save_base_path, dilate, animal_id, ncontrasts=1, regen=True, diff_mask=False):
    """
    Compiles images to be processed into a csv for radiomic processing

    Args:
        im_files (list of 3 str): path to images T1, T1C, T2
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

    # Get filenames
    image1T1 = im_files[0]
    image1T1c = im_files[1]
    image1T2 = im_files[2]
    label1 = mask_file

    # Get save names and ids
    simage1T1, _ = gen_savename(image1T1, save_base_path, save_type='T1')
    simage1T1c, _ = gen_savename(image1T1c, save_base_path, save_type='T1c')
    simage1T2, _ = gen_savename(image1T2, save_base_path, save_type='T2')
    slabel1, _ = gen_savename(label1, save_base_path, save_type='mask')

    # Rewrite to 16 bit
    if regen:
        print('Re-writing images as 16-bit')
        rewrite_nifti(image1T1, simage1T1)
        rewrite_nifti(image1T1c, simage1T1c)
        rewrite_nifti(image1T2, simage1T2)
        rewrite_nifti(label1, slabel1)

    # Dilate masks
    if (dilate != 0):
        slabel1 = dilate_masks(slabel1, simage1T2, dilate_kernel_size=dilate, diff=diff_mask)

    # Write CSV file
    csv_file = os.path.join(save_base_path, 'radiomics_files_multi.csv')
    write_csv(csv_file,
              3 * [animal_id], [simage1T1] + [simage1T1c] + [simage1T2],
              3*[slabel1])

    return csv_file


def write_csv(csv_file, id, image_file, mask_file):

    f = open(csv_file, 'w')
    f.write('ID,Image,Mask\n')

    for i in range(len(id)):
        write_str = id[i] + ',' + image_file[i] + ',' + mask_file[i] + '\n'
        f.write(write_str)


def dilate_masks(mask_files, im_files, dilate_kernel_size, diff=False):
    print('Dilating tumor masks')

    # Generate updated filename
    path, ext1 = os.path.splitext(mask_files)
    path, ext2 = os.path.splitext(path)
    path, filename = os.path.split(path)

    # Load file
    x_mask = nib.load(mask_files).get_data().astype(np.float16)
    x = nib.load(im_files).get_data().astype(np.float32)

    if diff:
        # Only use the difference between the original and dilated masks

        # Load already dilated bed mask
        bed_file = os.path.join(path, filename + '_bed' + ext2 + ext1)
        X_dia = nib.load(bed_file).get_data().astype('bool')

        # Take the difference
        X_dia = np.logical_xor(X_dia, x_mask)

        sfile = os.path.join(path, filename + '_edge' + ext2 + ext1)

    else:
        sfile = os.path.join(path, filename + '_bed' + ext2 + ext1)

        # Dilate the mask
        # selem = ball(dilate_kernel_size, dtype=np.float16)

        # xrange = np.arange(-dilate_kernel_size - 5, dilate_kernel_size + 6)
        # sigma = dilate_kernel_size/np.sqrt(-2 * np.log(1/2))
        # selem = np.exp(-xrange**2/(2 * sigma**2))
        # selem = selem/selem.sum()

        t = time()
        try:
            # X_dia = tf_dilation.compute(x_mask, selem)
            model = tf.keras.models.load_model('Crawler/dilation_model.h5')
            with tf.device('GPU:0'):
                X_dia = model.predict(x_mask[np.newaxis, :, :, :, np.newaxis])
            X_dia = X_dia.squeeze().astype('bool')

        except ValueError:
            print('Invalid input size for Tensorflow calculation, defaulting to Skimage functions')

            X_dia = binary_dilation(x_mask, selem)

        print('\tTime to dilate mask: %0.3f seconds' % ((time() - t)))

    # Filter out air
    inds = x < np.median(x)
    inds = binary_closing(inds, ball(3))
    X_dia[inds] = False

    print('\t%s' % sfile)

    # Save the file
    nib.save(nib.Nifti1Image(X_dia.astype(np.uint16), np.eye(4)), sfile)

    return sfile


class mask_dilation_class:
    def __init__(self, selem, gpu='GPU:0'):
        self.mask_in = []
        self.dia_graph = []
        self.gpu = gpu
        # self.mask_in = tf.placeholder(dtype=tf.float16, shape=(280, 280, 60), name='mask')
        # self.kern_in = tf.convert_to_tensor(selem, dtype=tf.float16)
        # tf.placeholder(dtype=tf.float16, shape=(51, 51, 51), name='kernel')
        # self.dia_graph = self.graph(self.mask_in, self.kern_in)

        # self.sess = tf.Session()

        self.model_name = 'Crawler/dilation_model.h5'

    def graph(self, mask, kernel):

        with tf.device(self.gpu):
            # Convert mask and kernel to tensors
            tf_mask = tf.convert_to_tensor(mask, dtype=tf.float16)
            tf_kernel = tf.convert_to_tensor(kernel, dtype=tf.float16)

            # Increase dimensions
            tf_mask = tf.expand_dims(tf_mask, axis=-1)
            tf_mask = tf.expand_dims(tf_mask, axis=0)
            tf_kernel = tf.expand_dims(tf_kernel, axis=-1)
            tf_kernel = tf.expand_dims(tf_kernel, axis=-1)
            # tf_kernel = tf.expand_dims(tf_kernel, axis=-1)

            # Convolve
            # tf_kernel_1 = tf.expand_dims(tf_kernel, axis=-1)
            # conv_dilated = tf.nn.convolution(tf_mask,
            #                                  tf_kernel_1,
            #                                  padding='SAME'
            #                                  )

            # tf_kernel_2 = tf.expand_dims(tf_kernel, axis=0)
            # conv_dilated = tf.nn.convolution(conv_dilated,
            #                                  tf_kernel_2,
            #                                  padding='SAME'
            #                                  )
            #
            # tf_kernel_3 = tf.expand_dims(tf_kernel, axis=1)
            # conv_dilated = tf.nn.convolution(conv_dilated,
            #                                  tf_kernel_3,
            #                                  padding='SAME'
            #                                  )

            conv_dilated = tf.nn.convolution(tf_mask,
                                             tf_kernel,
                                             padding='SAME')

            # Threshold
            cond = tf.greater(conv_dilated, 0.05)
            conv_dilated = tf.where(cond, tf.ones(tf.shape(conv_dilated)), tf.zeros(tf.shape(conv_dilated)))

            # Reduce dimensions
            dilated = tf.squeeze(conv_dilated)

            tf.keras.layers.InputLayer()

        return dilated

    def graph_keras(self, kernel):

        with tf.device(self.gpu):
            # Inputs
            x_input = tf.keras.layers.Input(shape=(None, None, None, 1), name='x_input')

            # Filter input image
            lp_input = tf.keras.layers.Conv3D(filters=1, padding='SAME', kernel_size=(kernel.shape[0],
                                                                                      kernel.shape[1],
                                                                                      kernel.shape[2]),
                                        use_bias=False, trainable=False, name='dilate')(x_input)

            # Assemble model
            model = tf.keras.models.Model(inputs=x_input, outputs=lp_input)

            # Set weights for non-trainable layer
            model.layers[1].set_weights([kernel])

            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

        model.save(self.model_name)

        return model

    def load_graph(self):

        return tf.keras.models.load_model(self.model_name)

    def compute(self, mask, kernel):

        # self.mask_in = tf.placeholder(dtype=tf.float16, shape=mask.shape, name='mask')
        # self.dia_graph = self.graph(self.mask_in, self.kern_in)

        # dia = self.sess.run(self.dia_graph, feed_dict={self.mask_in: mask})

        # dia = self.sess.run(self.dia_graph, feed_dict={self.mask_in: mask, self.kern_in: kernel})
        # dia = self.sess.run(self.graph(mask, kernel))

        # Keras
        kernel = kernel[:, :, :, np.newaxis, np.newaxis]
        mask = mask[np.newaxis, :, :, :, np.newaxis]

        if os.path.exists(self.model_name):
            model = self.load_graph()
        else:
            model = self.graph_keras(kernel)

        dia = model.predict(mask)
        dia = np.squeeze(dia)

        return dia.astype('bool')


def rewrite_nifti(files, sfiles):

    # Load file
    X = nib.load(files).get_data().astype(np.float32)

    # Normalize if image
    if len(np.unique(X)) > 3:
        X = (X - X.min()) / (X.max() - X.min())
        X = (2**16 - 1) * X

    X = X.astype(np.uint16)

    # Rewrite file
    nib.save(nib.Nifti1Image(X, np.eye(4)), sfiles)


def run_radiomics(outPath, inputCSV, outputFile):

    outputFilepath = outputFile
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')
    params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')

    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

    try:
        # Use pandas to read and transpose ('.T') the input data
        # The transposition is needed so that each column represents one test case. This is easier for iteration over
        # the input cases
        flists = pandas.read_csv(inputCSV).T
    except Exception:
        logger.error('CSV READ FAILED', exc_info=True)
        exit(-1)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists.columns))

    if os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeaturesExtractor(params)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3]
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True
        settings['normalize'] = True
        # settings['normalizeScale'] = 3
        settings['removeOutliers'] = 3

        extractor = featureextractor.RadiomicsFeaturesExtractor(**settings)
        # extractor.enableInputImages(wavelet= {'level': 2})

    logger.info('Enabled input images types: %s', extractor._enabledImagetypes)
    logger.info('Enabled features: %s', extractor._enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    # Instantiate a pandas data frame to hold the results of all patients
    results = pandas.DataFrame()

    for entry in flists:  # Loop over all columns (i.e. the test cases)
        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                    entry + 1,
                    len(flists),
                    flists[entry]['Image'],
                    flists[entry]['Mask'])

        imageFilepath = flists[entry]['Image']
        maskFilepath = flists[entry]['Mask']
        label = flists[entry].get('Label', None)

        if str(label).isdigit():
          label = int(label)
        else:
          label = None

        if (imageFilepath is not None) and (maskFilepath is not None):
            featureVector = flists[entry]  # This is a pandas Series
            featureVector['Image'] = os.path.basename(imageFilepath)
            featureVector['Mask'] = os.path.basename(maskFilepath)

        try:
            # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
            # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
            # as the values in the rows.
            result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
            featureVector = featureVector.append(result)
        except Exception:
            logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

        # To add the calculated features for this case to our data frame, the series must have a name (which will be the
        # name of the column.
        featureVector.name = entry
        # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
        # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
        # it is 'joined' with the empty data frame.
        results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

    logger.info('Extraction complete, writing CSV')
    # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
    results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
    logger.info('CSV writing complete')

    # Close out logging file
    x = list(rLogger.handlers)
    for i in x:
        rLogger.removeHandler(i)
        i.flush
        i.close()

    del logger, handler


def load_study_data(summary_file):

    # Read in study data
    df_control = pandas.read_excel(summary_file, skiprows=2, usecols=range(1, 22))
    df_pd1 = pandas.read_excel(summary_file, skiprows=2, usecols=range(24, 50))

    # Remove ".1", "#", and blank spaces from all keys
    remove_excess = lambda i: i.strip('.1').strip('#').strip()
    control_keys = [remove_excess(key) for key in df_control.keys()]
    pd1_keys = [remove_excess(key) for key in df_pd1.keys()]

    # Rename keys using the control group (which did not contain typos)
    key_dict = {key: control_keys[i] for i, key in enumerate(df_control.keys())}
    df_control = df_control.rename(index=str, columns=key_dict)

    key_dict = {key: control_keys[i] for i, key in enumerate(df_pd1.keys())}
    df_pd1 = df_pd1.rename(index=str, columns=key_dict)

    # Remove rows with all NaN values
    df_control = df_control.dropna(axis=0, how='all')
    df_pd1 = df_pd1.dropna(axis=0, how='all')

    # Add a Group column to the data
    df_control['Group'] = 'Control'
    df_pd1['Group'] = 'PD1'

    # Combine dataframes
    df = pandas.concat([df_control, df_pd1], ignore_index=True)

    return df


def init_dilation_class(dilation_rad):
    global tf_dilation

    selem = ball(dilation_rad, dtype=np.float16)

    tf_dilation = mask_dilation_class(selem)
