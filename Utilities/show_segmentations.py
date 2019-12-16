import os
import nibabel as nib
import numpy as np
from overlay_ims import overlay_image, save_overlay_im
from PIL import Image


def make_montage(sname_ims, sname_mont='montage.png', crop=None):
    """
    Make images
    Args:
        sname_ims (list): list of image names
        sname_mont (str):  name of output file
        crop (None or list of length 4): Crop coordinates (x, y, x_width, y_width)

    Returns:

    """

    rows = len(sname_ims)
    y = 0
    for i in range(rows):
        x = 0
        # Load first row of images
        im = Image.open(sname_ims[i])
        msk = Image.open(sname_msks[i])
        seg = Image.open(sname_segs[i])
        diff = Image.open(sname_diffs[i])

        # Apply crops
        if crop:
            bbox = (crop[0], crop[1], im.width + crop[2], im.height + crop[3])
            im = im.crop(bbox)
            msk = msk.crop(bbox)
            seg = seg.crop(bbox)
            diff = diff.crop(bbox)
            print(im.height)

        if i == 0:
            mont = Image.new('RGB', size=(im.width * 4, im.height * rows))

        mont.paste(im, (x, y))
        x += im.width
        mont.paste(msk, (x, y))
        x += im.width
        mont.paste(seg, (x, y))
        x += im.width
        mont.paste(diff, (x, y))

        # Increment row
        y += im.height

    # Save montage image
    sname_mont = os.path.join(save_path, sname_mont)
    mont.save(sname_mont)


def make_montage_init(sname_ims, sname_mont='montage.png'):
    """
    Make a montage with one row
    Args:
        sname_ims (str): montage save name
        sname_mont (list): lits of image files

    Returns:

    """
    # Make montage
    cols = len(sname_ims)
    x = 0
    y = 0
    for i in range(cols):
        # Load first row of images
        im = Image.open(sname_ims[i])

        if i == 0:
            mont = Image.new('RGB', size=(im.width * cols, im.height))

        mont.paste(im, (x, y))
        x += im.width

    # Save montage image
    sname_mont = os.path.join(save_path, sname_mont)
    mont.save(sname_mont)


def load_label_show(files, sz):
    """
    Load and prep label file
    Args:
        files (str): label filename
        sz (list of length 2): size of MR image (may not always match label)

    Returns:

    """
    from skimage.transform import resize
    from scipy.ndimage.morphology import binary_closing

    # Load image
    label = nib.load(files).get_data().astype('float32').squeeze()

    # Resize image
    label /= label.max()
    label = resize(label, (sz[0], sz[1]))

    # Make binary
    label[label < 0.5] = 0
    label[label > 0.5] = 1

    # Close the mask
    # label = binary_closing(label, structure=np.ones(shape=(3, 3, 3)))

    return 255 * label.astype(np.uint8)


"""
Goal: remake segmentation images to show a control and a PD1 animal before and after segmentation.
"""

# Set up save paths
save_path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/Analysis/Segmentation_images'

snames = [
    ['control_1',
     'control_2'],
    ['pd1_1',
     'pd1_2']
]

# Data paths
image_files = [
    ['/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520552/20180510/T2_cor.nii.gz',
     '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520552/20180517/T2_cor.nii.gz'],  # Control
    ['/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180503/T2_cor.nii.gz',
     '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180510/T2_cor.nii.gz']
]

seg_files = [
    ['/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520552/20180510/tumor_seg.nii.gz',
     '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520552/20180517/tumor_seg.nii.gz'],
    ['/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180503/tumor_seg.nii.gz',
     '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180510/tumor_seg.nii.gz']
]

mask_files = [
    ['/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520552/Seg_520552-1-label.nii',
     '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520552/Seg_520552-2-label.nii'],
    ['/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520457/Seg_520457-1-label.nii',
     '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520457/Seg_520457-2-label.nii']
]

slices = [[27, 19],
          [27, 19]
          ]

# Change paths if in windows
if 'nt' in os.name:
    save_path = save_path.replace('/media/matt/Seagate Expansion Drive/', 'E://')

    for i in range(len(image_files)):
        image_files[i] = [i.replace('/media/matt/Seagate Expansion Drive/', 'E://') for i in image_files[i]]
        seg_files[i] = [i.replace('/media/matt/Seagate Expansion Drive/', 'E://') for i in seg_files[i]]
        mask_files[i] = [i.replace('/media/matt/Seagate Expansion Drive/', 'E://') for i in mask_files[i]]

if not os.path.exists(save_path):
    os.mkdir(save_path)

# Colormap
colormap_mask = ['black', 'red']
colormap_seg = ['black', 'green']
colormap_comb = ['black', 'red', 'green']

# Histogram clipping
top_pct = 0.05 / 100
bot_pct = 15 / 100

sname_ims, sname_msks, sname_segs, sname_diffs = [], [], [], []
for i in range(len(image_files)):

    for ii in range(len(image_files[i])):
        # Read image
        im = nib.load(image_files[i][ii]).get_data().astype(np.float)
        msk = nib.load(mask_files[i][ii]).get_data().astype(np.float)
        seg = nib.load(seg_files[i][ii]).get_data().astype(np.float)

        # Scale the image
        im_s = sorted(im.reshape(-1))
        top_thresh = im_s[-round(len(im_s) * top_pct)]
        bot_thresh = im_s[round(len(im_s) * bot_pct)]
        im[im > top_thresh] = top_thresh
        im[im < bot_thresh] = bot_thresh

        # Select slice
        s = slices[i][ii]
        im = im[:, :, s].T
        msk = msk[:, :, s].T
        seg = seg[:, :, s].T

        # Normalize

        # Set up save name
        save_name = os.path.join(save_path, snames[i][ii])

        # Create overlay image - no masks
        sname_im = save_name + 'im.png'
        over_im = overlay_image(base_image=im,
                                overlay_image=np.zeros_like(msk),
                                colormap=colormap_mask,
                                sname=sname_im
                                )

        # Create overlay image - mask
        sname_msk = save_name + 'msk.png'
        over_im = overlay_image(base_image=im,
                                overlay_image=msk,
                                colormap=colormap_mask,
                                sname=sname_msk
                                )

        # Create overlay image - segmentation
        sname_seg = save_name + 'seg.png'
        over_im = overlay_image(base_image=im,
                                overlay_image=seg,
                                colormap=colormap_seg,
                                sname=sname_seg
                                )

        # Create overlay image - combined
        # Green = false positive, 1.0
        # Red = false negative, 0.5

        # False positives
        fp = seg - msk
        fp[fp < 0] = 0

        # False negatives
        fn = msk - seg
        fn[fn < 0] = 0

        # Combined
        comb = fp + 0.5 * fn

        sname_diff = save_name + 'diff.png'
        over_im = overlay_image(base_image=im,
                                overlay_image=comb,
                                colormap=colormap_comb,
                                sname=sname_diff
                                )

        sname_ims.append(sname_im)
        sname_msks.append(sname_msk)
        sname_segs.append(sname_seg)
        sname_diffs.append(sname_diff)

make_montage(sname_ims, crop=(35, 10, -35, -70))

""" Initial images """
# Show modal images and segmentation
files = ['/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/3/180420-3b-2_T1.nii',
         '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/3/180420-3b-2_T1wC.nii',
         '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/3/180420-3B-2_T2.nii']
ims = []
seg_file = '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/3/180420-3B-2-label.nii'
snames = ['T1', 'T1C', 'T2', 'seg']
snames = [os.path.join(save_path, i + '_2.png') for i in snames]
slice = 33

# Change paths if in windows
if 'nt' in os.name:
    seg_file = seg_file.replace('/media/matt/Seagate Expansion Drive/', 'E://')
    files = [i.replace('/media/matt/Seagate Expansion Drive/', 'E://') for i in files]

for file, sname in zip(files, snames[:-1]):
    # Load the file and select slice
    im = nib.load(file).get_data().astype(np.float)
    im = im[:, :, slice].T

    # Remove outliers
    im_s = sorted(im.reshape(-1))
    top_thresh = im_s[-round(len(im_s) * top_pct)]
    bot_thresh = im_s[round(len(im_s) * bot_pct)]
    im[im > top_thresh] = top_thresh
    im[im < bot_thresh] = bot_thresh

    # Convert to 8 bit
    im -= im.min()
    im /= im.max()
    im *= 255
    im = im.astype(np.uint8)

    # Save image
    save_overlay_im(im, sname)

# Save the segmenation image
seg = nib.load(seg_file).get_data().astype(np.uint8)
seg = seg[:, :, slice].T
seg -= seg.min()
seg = seg / seg.max()
seg *= 255
seg = seg.astype(np.uint8)

# Apply colormap
seg = np.repeat(seg[:, :, np.newaxis], 3, 2)
seg[:, :, 1:] = 0

# Save
save_overlay_im(seg, snames[-1])

# Save montage
make_montage_init(snames, sname_mont='mont_init1.png')

# Save second sample images
files = ['/media/matt/Seagate Expansion Drive/MR Data/SarcomaSegmentations/Mouse IV/Mouse IV T1 FLASH.nii',
         '/media/matt/Seagate Expansion Drive/MR Data/SarcomaSegmentations/Mouse IV/Mouse IV T1FLASH with Contrast.nii',
         '/media/matt/Seagate Expansion Drive/MR Data/SarcomaSegmentations/Mouse IV/Bias Corrected Images/Mouse IV T2TurboRARE New Bias Correction.nii']
seg_file = '/media/matt/Seagate Expansion Drive/MR Data/SarcomaSegmentations/Mouse IV/Mouse IV ROI Black and White Volume.nii'
snames = ['T1', 'T1C', 'T2', 'seg']
snames = [os.path.join(save_path, i + '.png') for i in snames]
slice = 25

# Change paths if in windows
if 'nt' in os.name:
    seg_file = seg_file.replace('/media/matt/Seagate Expansion Drive/', 'E://')
    files = [i.replace('/media/matt/Seagate Expansion Drive/', 'E://') for i in files]

for file, sname in zip(files, snames[:-1]):
    # Load the file and select slice
    im = nib.load(file).get_data().astype(np.float)
    sz = im.shape
    im = im[:, :, slice].T

    # Remove outliers
    im_s = sorted(im.reshape(-1))
    top_thresh = im_s[-round(len(im_s) * top_pct)]
    bot_thresh = im_s[round(len(im_s) * bot_pct)]
    im[im > top_thresh] = top_thresh
    im[im < bot_thresh] = bot_thresh

    # Convert to 8 bit
    im -= im.min()
    im /= im.max()
    im *= 255
    im = im.astype(np.uint8)

    # Save image
    save_overlay_im(im, sname)

# Save the segmenation image
seg = load_label_show(seg_file, sz)
seg = seg[:, :, slice].T

# Apply colormap
seg = np.repeat(seg[:, :, np.newaxis], 3, 2)
seg[:, :, 1:] = 0

# Save
save_overlay_im(seg, snames[-1])

# Save montage
make_montage_init(snames, sname_mont='mont_init2.png')

""" Show tumor, edge, and bed segmentations """
from Radiomics.radiomic_maps_190501 import dilate_masks

mask_tumor = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180510/tumor_seg.nii.gz'
image_name = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData/Results/K520457/20180510/T2_cor.nii.gz'
snames = ['area_tumor.png', 'area_edge.png', 'area_bed.png']
snames = [os.path.join(save_path, i) for i in snames]
descriptor = 'area'
slices = [29]

top_pct = 1 / 100
bot_pct = 15 / 100

# Compute masks
mask_bed_file = dilate_masks(mask_tumor, image_name, save_path, diff=False, name=descriptor)
mask_edge_file = dilate_masks(mask_tumor, image_name, save_path, diff=True, name=descriptor)

# Load image data
im = nib.load(image_name).get_data().astype(np.float32)
mask_tumor = nib.load(mask_tumor).get_data().astype(np.uint8)
mask_bed = nib.load(mask_bed_file).get_data().astype(np.uint8)
mask_edge = nib.load(mask_edge_file).get_data().astype(np.uint8)

# Delete dilated masks
# os.remove(mask_bed_file)
# os.remove(mask_edge_file)

# Preprocess image
# Remove outliers
# Remove by pct
im_s = sorted(im.reshape(-1))
top_thresh = im_s[-round(len(im_s) * top_pct)]
bot_thresh = im_s[round(len(im_s) * bot_pct)]
im[im > top_thresh] = top_thresh
im[im < bot_thresh] = bot_thresh

# Convert to 8 bit
# im -= im.min()
# im /= im.max()
# im *= 255
# im = im.astype(np.uint8)

colormap_seg2 = ['black', 'green']
rows = range(20, im.shape[0] - 20)
cols = range(10, im.shape[1] - 10)
# Make image overlays
for i, mask in enumerate([mask_tumor, mask_edge, mask_bed]):
    overlay_image(base_image=im[rows, :, slices[0]].T,
                  overlay_image=mask[rows, :, slices[0]].T,
                  colormap=colormap_seg2,
                  sname=snames[i])

# Combine images into a montage
make_montage_init(snames, sname_mont='area_montage.png')
