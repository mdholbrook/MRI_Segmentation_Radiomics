import numpy as np
from PIL import Image
from skimage import color
import nibabel as nib
from matplotlib.cm import get_cmap


def overlay_image(base_image, overlay_image, colormap, alpha=1.0, sname=None):
    """
    Creates a color overlay over in input image. Saves and returns the overlayed image.

    Args:
        base_image (2D numpy array): base image over which to overlay
        overlay_image (2D numpy array): image to overlay
        colormap (str or list): string containing standard matplotlib colormaps.
            Also can be a list of colors for custom colormaps.
        alpha (float): overlay transparency
        sname (str): Optional, filename for saving the overlayed image

    Returns:

    """

    # Apply colormap to overlay image
    overlay_image, min_val = apply_cm(overlay_image, colormap)

    # Overlay image
    overlayed_image = im_overlay(base_image, overlay_image, min_val, alpha=alpha)

    # Save the image if a name was given
    if sname:
        save_overlay_im(overlayed_image, sname)

    return overlayed_image


def apply_cm(over_im, cm):
    """
    Applied a specific colormap to the image which will be overlayed
    Args:
        over_im (numpy array):
        cm (str or list): matplotlib colormap

    Returns:
        (numpy array): image with new colormap
    """

    # Check input colormap - if it is a list create a custom map
    if type(cm) == list:
        from matplotlib.colors import LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list('mycmap', cm)
    elif type(cm) == str:
        cm = get_cmap(cm)

    # Normalize
    # if not (over_im.min() == 0) and not (over_im.max() == 1):
    #     over_im -= over_im.min()
    #     over_im /= over_im.max()

    # Apply colormap
    im = cm(over_im)

    # Get min value
    min_val = cm(0)[:3]

    # Convert to RGB from RGBA
    return im[:, :, :3], min_val


def im_overlay(mr_im, over_im, min_val, alpha=1.0):
    """
    Combine the MR image and the overlay.
    Args:
        mr_im (numpy array): MR image
        over_im (nupy array): overlay image
        min_val (float): removes this value from the overlay
        alpha (float): transparency of the overlay

    Returns:
        (numpy array): combined image
    """

    # Compress images to 2D
    mr_im = mr_im.squeeze()
    over_im = over_im.squeeze()

    # Normalize
    mr_im -= mr_im.min()
    mr_im /= mr_im.max()

    # Remove the background from the mask
    # over_im = np.ma.masked_where(over_im == 0, over_im)

    # https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
    # Convert images to RGB
    im_rep = np.dstack((mr_im, mr_im, mr_im))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    over_hsv = color.rgb2hsv(over_im)
    mr_hsv = color.rgb2hsv(im_rep)

    mask = ((over_im[:, :, 0] == min_val[0]) &
            (over_im[:, :, 1] == min_val[1]) &
            (over_im[:, :, 2] == min_val[2]))
    mask = ~mask

    # Replace the hue and saturation of the original image
    # with that of the color mask
    mr_hsv[mask, 0] = over_hsv[mask, 0]
    mr_hsv[mask, 1] = over_hsv[mask, 1] * alpha

    # Set up mask - remove image data where there is mask data
    if alpha > 1:
        mr_hsv[mask, :] = over_hsv[mask, :]

    # Convert bach to RGB
    overlayed_im = color.hsv2rgb(mr_hsv)

    # Remove outliers
    bit16 = 2**16 - 1
    thresh = 0.07
    overlayed_im -= overlayed_im.min()
    overlayed_im /= overlayed_im.max()
    overlayed_im *= bit16
    overlayed_im[overlayed_im < (bit16 * thresh)] = bit16 * thresh
    overlayed_im[overlayed_im > (bit16 * 0.97)] = bit16 * 0.97

    # Convert image to 8-bit
    # overlayed_im[overlayed_im < overlayed_im.min() * 1.5] = 0
    overlayed_im -= overlayed_im.min()
    overlayed_im /= overlayed_im.max()
    overlayed_im *= 255
    overlayed_im = overlayed_im.astype(np.uint8)

    return overlayed_im


def save_overlay_im(over_im, sname):

    # Save as image using PIL
    pil_im = Image.fromarray(over_im)
    pil_im.save(sname)


if __name__ == '__main__':
    """
    Example of how to use this code to overlay images
    """

    file1 = '/media/matt/Seagate Expansion Drive/b7TData/Results/K520503/20180503/T2_cor.nii.gz'
    file2 = '/media/matt/Seagate Expansion Drive/b7TData/Results/K520503/20180503/tumor_seg.nii.gz'

    mr_im = nib.load(file1).get_data().astype(np.float)
    over_im = nib.load(file2).get_data().astype(np.float)

    mr_im = mr_im[:, :, 30].T
    over_im = over_im[:, :, 30].T

    from skimage.filters import gaussian
    over_im = gaussian(over_im, sigma=3)
    over_im[over_im < 0.1] = 0

    # Apply colormap to mask
    cm = 'jet'
    # cm = get_cmap('brg')
    # cm = ['black', 'green']

    over_im, min_val = apply_cm(over_im, cm)

    im_over = im_overlay(mr_im, over_im, min_val, alpha=1)

    save_overlay_im(im_over, 'overlay.png')
