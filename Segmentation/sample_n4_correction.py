import os
import subprocess
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
import nibabel as nib
import numpy as np

"""
Test file for bias correction using N4BiasFieldCorrection from ITK
"""

# T2 = '/home/matt/Documents/Test_bias_cor/Mouse_IV_T2TurboRARE.nii'
# T2_out = '/home/matt/Documents/Test_bias_cor/Mouse_IV_T2TurboRARE_pyCorr.nii'
# weight_im = '/home/matt/Documents/Test_bias_cor/Mouse_IV_T1_FLASH.nii'
# bias_im = '/home/matt/Documents/Test_bias_cor/bias.nii'

T2 = '/home/matt/Documents/BiasCorrectionTest/520552_D1_T2.nii'
T2_out = '/home/matt/Documents/BiasCorrectionTest/T2_pyCorr.nii'
weight_im = '/home/matt/Documents/BiasCorrectionTest/520552_D1_T1.nii'
bias_out = '/home/matt/Documents/BiasCorrectionTest/bias.nii'

# Load Slicer output
bias_slc = '/home/matt/Documents/BiasCorrectionTest/520522_D1_BiasField.nii'
T2_slc = '/home/matt/Documents/BiasCorrectionTest/520552_D1_T2BC.nii'
bias_slc = nib.load(bias_slc).get_data().astype('float32')
T2_slc = nib.load(T2_slc).get_data().astype('float32')

# Perform correction
bias_er = []
T2_er = []
params = range(1, 2)
for i in params:
    # n4 = N4BiasFieldCorrection()
    # n4.inputs.input_image = T2
    # n4.inputs.save_bias = True
    # n4.inputs.copy_header = False
    # n4.inputs.dimension = 3
    # n4.inputs.mask_image = None
    # n4.inputs.weight_image = weight_im
    # n4.inputs.output_image = T2_out
    # n4.inputs.bspline_fitting_distance = i
    # n4.inputs.bspline_order = 3
    # n4.inputs.shrink_factor = 4
    # n4.inputs.n_iterations = [100, 100, 100]
    # n4.inputs.convergence_threshold = 0.005
    # n4.inputs.bias_image = bias_out
    # n4.inputs.num_threads = 7

    cmd = 'N4BiasFieldCorrection ' \
          '--bspline-fitting [ 1x1x1, 3 ] ' \
          '-d 3 ' \
          '--input-image %s ' \
          '--convergence [ 100x100x100, 0.005 ] ' \
          '--output [ %s, %s ] ' \
          '--shrink-factor 4 ' \
          '--weight-image %s ' \
          '--histogram-sharpening [0.3, 0.01, 200]' % (T2, T2_out, bias_out, weight_im)

    # os.system(cmd)
    # n4.run()
    subprocess.Popen(cmd, shell=True).wait()

    # Compare Python outputs to Slicer
    bias_py = nib.load(bias_out).get_data().astype('float32')
    T2_py = nib.load(T2_out).get_data().astype('float32')

    # Compute rmse
    bias_er.append(np.sqrt(np.sum((bias_slc.reshape(-1) - bias_py.reshape(-1))**2)))
    T2_er.append(np.sqrt(np.sum((T2_slc.reshape(-1) - T2_py.reshape(-1))**2)))

for i in range(len(params)):
    print('\nSpline fitting distance: %d' % params[i])
    print('\tT2 RMSE: %0.3f' % T2_er[i])
    print('\tBias RMSE: %0.3f' % bias_er[i])
