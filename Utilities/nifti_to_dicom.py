import os
import nibabel as nib
import pydicom as dcm
import numpy as np


def txt_to_dcm(txt_file):
    """
    Read a text file containing dicom header information
    Args:
        txt_file (str): path to file
            eg. txt_file = '/media/matt/Seagate Expansion Drive/b7TData/Results/K520457/20180510/dicom_header.txt'

    Returns:
        dict of dicom headers
    """

    newdict = {}

    f = open(txt_file, 'r')

    for line in f:
        listedline = line.strip().split('  ')

        if len(listedline) > 1:  # New line

            listdata = [x for x in listedline if x != '']

            if len(listdata) == 2:  # Line contains data

                tag = listdata[0].split(') ')[1]

                vr, value = listdata[1].split(': ')

                newdict[tag] = [vr, value]

    f.close()

    # Set up DICOM header
    dm = dcm.Dataset()

    # Populate from txt file
    for key in newdict.keys():

        tag = key
        vr = newdict[key][0]
        value = newdict[key][1]

        dm.add_new(tag, vr, value)

    return newdict


if __name__ == "__main__":
    """
    This sample code reads in nifti files and converts them to dicoms
    """

    # Source data paths
    nifti_files = ['/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520457/520457-1_T1.nii',
                   '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520457/520457-1_T1wC.nii',
                   '/media/matt/Seagate Expansion Drive/MR Data/MR_Images_Sarcoma/520457/520457-1_T2.nii'
                   ]

    # Template dicom data - T1, T1C, T2
    sample_dcms = ['/media/matt/Seagate Expansion Drive/b7TData/20180510_102653_B20035_1_1/3/pdata/1/dicom/MRIm01.dcm',
                   '/media/matt/Seagate Expansion Drive/b7TData/20180510_102653_B20035_1_1/5/pdata/1/dicom/MRIm01.dcm',
                   '/media/matt/Seagate Expansion Drive/b7TData/20180510_102653_B20035_1_1/4/pdata/1/dicom/MRIm01.dcm'
                   ]

    # Save paths
    base_path = '/media/matt/Seagate Expansion Drive/b7TData/520457'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    save_paths = [os.path.join(base_path, 'T1'),
                  os.path.join(base_path, 'T1C'),
                  os.path.join(base_path, 'T2')]

    # Updates to dicom fields
    date = [['StudyDate', 'SeriesDate', 'AcquisitionDate'], '20180503']
    description = [['StudyDescription'], '180503-2\nDAY 1 of Scanning']

    for nifti_file, sample_dcm, save_path in zip(nifti_files, sample_dcms, save_paths):

        # Create save folder
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Load sample dicom
        d = dcm.read_file(sample_dcm)

        # Load Nifti
        x = nib.load(nifti_file).get_data().astype(np.float32)

        # Update dicom fields
        d.StudyDate = date[1]
        d.SeriesDate = date[1]
        d.AcquisitionDate = date[1]
        d.StudyDescription = description[1]

        # Get initial slice position and thickness
        slc_thickness = float(d.SliceThickness)
        init_pos = float(d.SliceLocation)

        # Update image data and write dicom
        for i in range(x.shape[2]):

            # Update slice location
            pos = init_pos + slc_thickness * i
            d.SliceLocation = "%0.2f" % pos

            # Get pixel information
            slc_pix = x[:, :, i].astype(np.int16).T

            # Update pixel information
            d.PixelData = slc_pix.tobytes()

            # Write dicom to file
            dname = os.path.join(save_path, 'MRIm%0.2d.dcm' % i)
            dcm.dcmwrite(dname, d)
