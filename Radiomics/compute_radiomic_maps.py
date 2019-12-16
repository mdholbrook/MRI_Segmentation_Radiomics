import os
from glob2 import glob
import numpy as np
from time import time
import SimpleITK as sitk
from radiomics import featureextractor
from Crawler.crawler_radiomics import gen_images_csv
import nibabel as nib
import tensorflow as tf
from PIL import Image
from skimage.morphology import selem, binary_dilation, binary_closing, ball
from overlay_ims import overlay_image


def make_montage(in_image, in_map, save_path, sname_mont='montage.png'):

    # Get input image shape
    sz = in_image.shape

    # Normalize map image
    in_map = (in_map - in_map.min()) / (in_map.max() - in_map.min())

    # Make montage
    rows = 3
    cols = 5
    inds = np.linspace(0, sz[0], rows*cols + 2).astype(int)[1:-1]
    y = 0
    n = 0
    for ii in range(rows):
        x = 0
        for i in range(cols):
            # Load first row of images
            im = in_image[inds[n]]
            msk = in_map[inds[n]]

            # Create overlay
            im = overlay_image(im, msk, colormap='autumn')

            # Convert to PIL Image
            im = Image.fromarray(im)

            if n == 0:
                mont = Image.new('RGB', size=(im.width*cols, im.height*rows))

            # Paste image into montage
            mont.paste(im, (x, y, x + im.width, y + im.height))
            x += im.width
            n += 1

        # Increment row
        y += im.height

    # Save montage image
    sname_mont = os.path.join(save_path, sname_mont)
    mont.save(sname_mont)


def dilate_masks(mask_files, im_files, outpath, diff=False, name=''):
    """
    Dilates binary image masks
    Args:
        mask_files (str): path to the image mask
        im_files (str): path to the image
        outpath (str): path in which to save output
        diff (bool): whether to use the difference between the tumor mask and the dilated mask (ei. only use the
        dilated area).
        name (str): an optional addition to the naming convention

    Returns:
        (str): path to dilated image mask
    """
    print('Dilating tumor masks')

    # Generate updated filename
    path, ext1 = os.path.splitext(mask_files)
    path, ext2 = os.path.splitext(path)
    path, filename = os.path.split(path)

    # Load file
    x_mask = nib.load(mask_files).get_data().astype(np.float16)
    x = nib.load(im_files).get_data().astype(np.float32)

    # Set flags
    new_mask_flag = False

    if diff:
        # Only use the difference between the original and dilated masks
        sfile = os.path.join(outpath, filename + '_edge_' + name + ext2 + ext1)

        if not os.path.exists(sfile):

            # Load already dilated bed mask
            bed_file = os.path.join(outpath, filename + '_bed_' + name + ext2 + ext1)
            X_dia = nib.load(bed_file).get_data().astype('bool')

            # Take the difference
            X_dia = np.logical_xor(X_dia, x_mask)

            sfile = os.path.join(outpath, filename + '_edge_' + name + ext2 + ext1)

            new_mask_flag = True

    else:
        sfile = os.path.join(outpath, filename + '_bed_' + name + ext2 + ext1)

        if not os.path.exists(sfile):

            t = time()
            try:
                # X_dia = tf_dilation.compute(x_mask, selem)
                model_name = '/home/matt/Documents/SegSarcoma/Crawler/dilation_model.h5'
                model = tf.keras.models.load_model(model_name)
                with tf.device('GPU:0'):
                    X_dia = model.predict(x_mask[np.newaxis, :, :, :, np.newaxis])
                X_dia = X_dia.squeeze().astype('bool')

            except ValueError:
                print('Invalid input size for Tensorflow calculation, defaulting to Skimage functions')

                X_dia = binary_dilation(x_mask, selem)

            print('\tTime to dilate mask: %0.3f seconds' % ((time() - t)))
            new_mask_flag = True

    if new_mask_flag:
        # Filter out air
        inds = x < np.median(x)
        inds = binary_closing(inds, ball(3))
        X_dia[inds] = False

        # Save the file
        nib.save(nib.Nifti1Image(X_dia.astype(np.uint16), np.eye(4)), sfile)

    print('\t%s' % sfile)

    return sfile


def compute_radiomics(im_path, outpath):
    """
    Compute new voxel-based radiomic maps.
    Args:
        im_path (str): path to multi-modal image files
        outpath (str): path in which to save results

    Returns:

    """

    # Set up save directory
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Get image filenames
    T1_file = os.path.join(im_path, 'T1.nii.gz')
    T1c_file = os.path.join(im_path, 'T1c.nii.gz')
    T2_file = os.path.join(im_path, 'T2_cor.nii.gz')
    mask_file = os.path.join(im_path, 'tumor_seg.nii.gz')

    # Working directory
    save_base_path = os.path.join(os.getcwd(), 'Working')
    if not os.path.exists(save_base_path): os.mkdir(save_base_path)

    # Get animal id
    base, _ = os.path.split(im_path)
    _, animal_id = os.path.split(base)

    # Constants
    dilate = 25

    # Set up values for multiple mask configurations
    sfiles = ['radiomic_features',
              'radiomic_features_bed',
              'radiomic_features_edge']
    dilate = [0, dilate, dilate]
    diff_mask = [False, False, True]

    # Set up empty list for radiomics files
    for i in range(3):
        if i == 0:
            regen = True
        else:
            regen = False

        # Append the current radiomics file
        radiomics_sfile = os.path.join(outpath, sfiles[i] + 'csv')
        radiomics_imfile = os.path.join(outpath, sfiles[i] + 'nii')

        # Generate CSV file of images/masks and re-save images as 16 bit
        csv_file = gen_images_csv([T1_file, T1c_file, T2_file],
                                  mask_file=mask_file,
                                  save_base_path=save_base_path,
                                  dilate=dilate[i],
                                  ncontrasts=3,
                                  regen=regen,
                                  diff_mask=diff_mask[i],
                                  animal_id=animal_id)

        # Run radiomics
        compute_radiomic_maps(outpath, T2_file, mask_file, feature_dict)


def compute_radiomic_maps(outpath, image_name, mask_name, feature_dict, descriptor, index):
    """
    Computes radiomic voxel-maps for individual features and images.
    Args:
        outpath (str): path to save outputs
        image_name (str): filename of image file
        mask_name (str): filename of mask file
        feature_dict (dict): dictionary of classes and features.
            E.g {'firstorder': ['Energy', ...], ...}

    Returns:

    """

    # Save image
    im = sitk.ReadImage(image_name)
    if index == 0:
        sname = os.path.join(outpath, '{}_{}_Image.nii'.format(descriptor, index))
        sitk.WriteImage(im, sname)

    # Set up params
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3]
    settings['interpolator'] = sitk.sitkBSpline
    settings['enableCExtensions'] = True
    settings['normalize'] = True
    settings['removeOutliers'] = 3

    # Voxel-based settings
    settings['kernelRadius'] = 3
    settings['initValue'] = 0
    settings['voxelBatch'] = -1

    # Instantiate the feature extractor class
    extractor = featureextractor.RadiomicsFeaturesExtractor(**settings)

    # Select features to compute
    extractor.enableAllFeatures()
    extractor.disableAllFeatures()
    for key, item in list(feature_dict.items()):
        extractor._enabledFeatures[key] = item

    # Calculate the voxel-based features
    t = time()
    result = extractor.execute(image_name, mask_name, voxelBased=True)
    print('\t\t%0.2f seconds to extract %d features' % (time() - t, len(feature_dict.keys())))
    for key, val in list(result.items()):
        if isinstance(val, sitk.Image):  # Feature map
            # sname = os.path.join(outpath, key + '_{}.nii.gz'.format(descriptor))
            # sitk.WriteImage(val, sname, True)
            keys = key.split('_')
            key = keys[1] + '_' + keys[2]

            val_sz = val.GetSize()
            val_np = sitk.GetArrayFromImage(val)

            # Rewrite as the same size as the input image and matching the mask
            label = sitk.ReadImage(mask_name)
            sz = label.GetSize()
            sz = [sz[1:], sz[0]]
            label = sitk.GetArrayFromImage(label)

            # Get x, y, z starting indicies
            sum_x = np.sum(np.sum(label, axis=-1), axis=-1)
            first_x = np.where(sum_x > 0)[0][0]

            sum_y = np.sum(np.sum(label, axis=0), axis=-1)
            first_y = np.where(sum_y > 0)[0][0]

            sum_z = np.sum(np.sum(label, axis=0), axis=0)
            first_z = np.where(sum_z > 0)[0][0]

            # Insert radiomic voxel map
            label_w = np.zeros_like(label)
            label_w[first_x:first_x+val_sz[2], first_y:first_y+val_sz[1], first_z:first_z+val_sz[0]] = val_np

            # Save full sized map
            label_w_im = sitk.GetImageFromArray(label_w)
            sname = os.path.join(outpath, '{}_{}_{}.nii.gz'.format(descriptor, index, key))
            sitk.WriteImage(label_w_im, sname)
            print("\t\tStored feature %s in %s\n\n" % (key, sname))

            # Save montage for ease of viewing
            im = sitk.GetArrayFromImage(im)
            make_montage(im, label_w, outpath, sname_mont='{}_{}_montage.png'.format(descriptor, index))

        else:  # Diagnostic information
            print("\t\t%s: %s" % (key, val))


def run_all(outpath, image_path, feature_list, descriptor):

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Get image names
    image_names = glob(os.path.join(image_path, 'T*.nii.gz'))
    image_names_copy = image_names.copy()

    # Filter out incorrectly filtered ones
    for image_name in image_names_copy:

        if 'T1c_cor' in image_name:
            image_names.remove(image_name)

        elif 'T1_cor' in image_name:
            image_names.remove(image_name)

        elif 'T2.' in image_name:
            image_names.remove(image_name)

        elif 'tumor' in image_name:
            mask_name = image_name
            image_names.remove(image_name)

    image_names = sorted(image_names)

    # Generate bed and edge masks - Tumor, bed, edge
    mask_names = list([])
    mask_names.append(mask_name)
    mask_names.append(dilate_masks(mask_name, image_names[-1], outpath, diff=False, name=descriptor))
    mask_names.append(dilate_masks(mask_name, image_names[-1], outpath, diff=True, name=descriptor))

    # Prepare to compute radiomic maps
    num_samps = len(feature_list)

    for i in range(num_samps):

        # Get run specifics
        contrast = feature_list[i][0]
        mask_area = feature_list[i][1]
        feature_dict = feature_list[i][2]

        # Get specific filenames
        if 'T1.nii' in contrast:
            image_name = image_names[0]
        elif 'T1c.nii' in contrast:
            image_name = image_names[1]
        else:
            image_name = image_names[2]

        if 'tumor' in mask_area:
            mask_name = mask_names[0]
        elif 'bed' in mask_area:
            mask_name = mask_names[1]
        else:
            mask_name = mask_names[2]

        print('Computing radiomic map for %s in the %s' % (contrast, mask_area))
        print('\t%s: %s' % (list(feature_dict.items())[0][0], list(feature_dict.items())[0][1][0]))

        # Determine if the data has already been processed
        if os.path.exists(os.path.join(outpath, '{}_{}_montage.png'.format(descriptor, i))):
            print('\t\tSet already processed\n\n')

        else:
            compute_radiomic_maps(outpath, image_name, mask_name, feature_dict, descriptor, index=i)


if __name__ == "__main__":

    """ Pre/Post RT comparison """
    # Pre RT
    outpath = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/Analysis/VoxelRadiomics'
    image_path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180503'
    descriptor = 'Pre'

    feature_list = [
        ['T2',  'tumor',    {'glszm': ['ZoneVariance']}],
        ['T1C', 'tumor',    {'glrlm': ['RunLengthNonUniformity']}],
        ['T1C', 'tumor',    {'glszm': ['LargeAreaLowGrayLevelEmphasis']}],
        ['T1C', 'tumor',    {'gldm': ['DependenceNonUniformity']}],
        # ['T2',  'bed',      {'gldm': ['DependenceNonUniformity']}],
        # ['T2',  'bed',      {'firstorder': ['Energy']}],
        # ['T1C', 'bed',      {'glrlm': ['RunVariance']}],
        # ['T1C', 'bed',      {'glrlm': ['LongRunHighGrayLevelEmphasis']}],
        # ['T1C', 'bed',      {'firstorder': ['Range']}],
        ['T2',  'edge',     {'firstorder': ['Energy']}],
        ['T1C', 'edge',     {'glrlm': ['GrayLevelNonUniformity']}]
        ]

    run_all(outpath, image_path, feature_list, descriptor)

    # Post RT
    image_path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180510'
    descriptor = 'Post'

    run_all(outpath, image_path, feature_list, descriptor)

    """ Recurrence comparison """
    # Recurrence
    image_path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K521297/20181206'
    descriptor = 'RecPost1'

    feature_list = [
        ['T2', 'edge',      {'glrlm': ['RunLengthNonUniformity']}],
        ['T1', 'edge',      {'glrlm': ['HighGrayLevelRunEmphasis']}],
        ['T1', 'edge',      {'glrlm': ['LowGrayLevelRunEmphasis']}],
        ['T2', 'edge',      {'ngtdm': ['Busyness']}],
        ]

    run_all(outpath, image_path, feature_list, descriptor)

    # No recurrence
    image_path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K521234/20181126'
    descriptor = 'NoRecPost1'

    run_all(outpath, image_path, feature_list, descriptor)

    # Recurrence T2
    # Recurrence
    image_path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K521297/20181206'
    descriptor = 'RecPost2'

    run_all(outpath, image_path, feature_list, descriptor)

    # No recurrence
    image_name = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K521234/20181126'
    descriptor = 'NoRecPost2'

    run_all(outpath, image_path, feature_list, descriptor)
