import os
import subprocess
from glob2 import glob
from pydicom import dcmread
import numpy as np
from time import time
import json
import jsbeautifier
from datetime import datetime
import re
import nibabel as nib
from Crawler.crawler_segmentation import seg_from_model, display_segmentations
from Crawler.crawler_radiomics import gen_images_csv, run_radiomics, load_study_data, init_dilation_class


def load_log_file(log_file):

    # Load database
    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)

    else:
        log = {}

    return log


class DataLoader:
    def __init__(self, base_path, folder_regex, log_file):
        self.base_path = base_path
        self.folder_regex = folder_regex
        self.log_file = log_file
        self.ind = 0
        self.current_files = False
        self.patient_name = []
        self.study_date = []
        self.acq_time = []
        self.protocol = []
        self.next_folder = ''

        print('Searching for data on the NAS')
        t1 = time()
        # Get folder names
        walker = [x[0] for x in os.walk(base_path)]
        walker.remove(base_path)

        # Get subdirectory names
        subdirs = [os.path.split(i) for i in walker]

        # # Select folders which match the regex
        # image_paths = [name for name in subdirs
        #                if name[1].lower().find(folder_regex.lower()) != -1]
        image_paths = []
        for name in subdirs:

            files = glob(os.path.join(os.path.join(name[0], name[1]), folder_regex))

            if files:
                image_paths.append(name)

        self.subdirs = [os.path.join(p[0], p[1]) for p in image_paths]

        # Sort files by date (oldest to newest)
        self.subdirs.sort()

        # Bundle scans
        self._bundle_scans()

        print('\tTime to build folder structure: %0.2f seconds' % (time() - t1))

        # Load log file
        print('Loading log file\n\t%s' % self.log_file)
        self.log = load_log_file(log_file)

    def _bundle_scans(self):
        """
        Bundles directories together by study using similar base folders.
        Returns:

        """

        # Set up used index
        bases = []
        bundles = []

        for i in range(len(self.subdirs)):

            # Get base directory
            bs = self.subdirs[i]
            flag = True
            while flag:
                bs1 = os.path.split(bs)[0]

                if bs1 == self.base_path.rstrip('/').rstrip('\\'):
                    flag = False
                else:
                    bs = bs1

            # Proceed if the base name as not been processed
            if bs not in bases:

                bases.append(bs)

                # Find base path matches
                matches = [k for (k, name) in enumerate(self.subdirs) if re.findall(bs, name)]

                # Fill in indexes and create output vector
                bundles.append([self.subdirs[f] for f in matches])

        # Update subdirectories
        self.subdirs = bundles

    def get_folder(self):
        """
        Pulls the next folder containing Dicom data
        Returns:
            (str): path to a folder containing Dicom files
        """

        if self.ind == len(self.subdirs):

            folder_exists = False

        else:

            self.next_folder = self.subdirs[self.ind]
            self.ind += 1
            self.current_files = False
            folder_exists = True

        return folder_exists

    def load_study_data(self):
        """
        Loads the patient name and study date using the data's Dicom header.
        Returns:

        """

        # Get filenames
        if not self.current_files:
            self._get_dicom_files()

        self.patient_name = []
        self.study_date = []
        self.acq_time = []
        self.protocol = []

        # Return animal ID (K-number, found in PatientName)
        for i in range(len(self.names)):
            dcm = dcmread(self.names[i][0])

            # Get and decode animal ID (K-number, found in PatientName)
            self.patient_name = dcm.PatientName.original_string.decode()
            self.patient_name = self.patient_name.strip('^')  # Some animals have following ^^^^
            self.patient_name = 'K' + self.patient_name

            # Get study date and time
            self.study_date = dcm.StudyDate
            self.acq_time.append(dcm.AcquisitionTime)

            # Get protocol data
            self.protocol.append(dcm.ProtocolName)

        return [self.patient_name, self.study_date]

    def load_dicom(self):
        """
        Loads Dicom image data from the last returned directory

        Returns:
            numpy array: a 3D image volume
        """

        if not self.current_files:
            self._get_dicom_files()

        # Sort by image protocol
        sort_inds = self._sort_protocols()

        # For each of the three contrasts
        for i in range(len(self.names)):

            # Get index corresponding to the correct contrast (T1, T1c, T2)
            ind = sort_inds[i]

            if i == 0:
                # Load Dicom data
                dcm = dcmread(self.names[i][0])
                dims = (dcm.Rows, dcm.Columns, len(self.names[0]), len(self.names))
                im_vol = np.zeros(shape=dims)

            for (ii, name) in enumerate(self.names[i]):

                # Load Dicom
                dcm = dcmread(name)

                # Load files as a numpy array
                im_vol[:, :, ii, ind] = dcm.pixel_array

        self.current_files = False

        return im_vol

    def compare_with_log(self):
        """
        Log data is a dictionary with fields: 'AnimalID', 'Scan1', 'Scan2'
        Returns:
            bool: True if the animal and scan date appear in the processed log

        """

        # Initialize variables
        animal_scanned = False
        date_scanned = False

        if self.patient_name in self.log.keys():  # If the animal exists in the DB

            animal_scanned = True

            if self.study_date in self.log[self.patient_name]['StudyDate']:  # If the scan date exists for the animal

                date_scanned = True

        already_processed = animal_scanned and date_scanned

        if already_processed:

            print('\t\tScan already processed!')

        return already_processed

    def _get_dicom_files(self, file_regexp='*.dcm'):

        self.names = []

        # For each contrast
        print('Processing:')
        for c in range(3):
            image_path = os.path.join(self.next_folder[c], file_regexp)
            print('\t %s' % self.next_folder[c])

            # Get file names
            self.names.append(glob(image_path))
            self.names[c].sort()

        # Flag file list as current
        self.current_files = True

    def _sort_protocols(self):
        """
        Sort scans so that the data can be returned as T1, T1 with contrast, and T2.
        Returns:

        """

        # Initialize the index for T2 = 2, only need to rearrange T1 contrasts
        sort_inds = [2 for _ in range(len(self.names))]
        times = []

        # Collect protocol and time data
        for i in range(len(self.names)):

            if re.findall('T1', self.protocol[i]):
                sort_inds[i] = 1

            times.append(datetime.strptime(self.acq_time[i], '%H%M%S'))

        # Sort T1 scans by time
        t = [(i, times[i]) for (i, ind) in enumerate(sort_inds) if ind == 1]

        if t[0][1] < t[1][1]:  # if the first index happened first
            sort_inds[t[0][0]] = 0
        else:
            sort_inds[t[1][0]] = 0

        return sort_inds


class SaveResults:
    def __init__(self, data_base_path, save_folder, log_file):

        # Initialize variables
        self.animal_folder = None
        self.curr_scan_path = None
        self.snames = []

        # Set up paths
        self.base_save_path = os.path.join(data_base_path, save_folder)
        self.log_file = log_file

        if not os.path.exists(self.base_save_path):
            os.mkdir(self.base_save_path)

        # Load log file
        self.log = load_log_file(log_file)

    def append_to_log(self, animal_id, study_date):
        """
        Appends the animal and study date to the log of processed studies
        Args:
            animal_id (str): The animal K-number
            study_date (str): The study date (YYYYMMDD)

        Returns:

        """

        if animal_id in self.log.keys():

            # Animal has been scanned before
            self.log[animal_id]['StudyDate'].append(study_date)

        else:

            # New entry
            self.log[animal_id] = {'StudyDate': [study_date]}

    def save_log(self):
        """
        Saves the dictionary detailing which sets have been processed
        Returns:

        """

        # Reformat the json file for easier reading
        formatted_log = jsbeautifier.beautify(json.dumps(self.log))

        with open(self.log_file, 'w') as f:
            f.write(formatted_log)

    def gen_save_path(self, animal_id, study_date):
        """
        Generates output folders for analyzed data.
        Args:
            animal_id (str): The animal K-number
            study_date (str): The study date (YYYYMMDD)

        Returns:
            str: the base path for the animal
            str: the study date path
        """

        # Generate animal folder
        animal_folder = os.path.join(self.base_save_path, animal_id)

        # Make the animal folder if it does not exist
        if not os.path.exists(animal_folder):
            os.mkdir(animal_folder)

        # Set up current animal save path
        curr_animal_path = os.path.join(animal_folder, study_date)

        if not os.path.exists(curr_animal_path):
            os.mkdir(curr_animal_path)

        self.animal_folder = animal_folder
        self.curr_scan_path = curr_animal_path

        # Set up save names
        snames = ['T1.nii.gz', 'T1c.nii.gz', 'T2.nii.gz']
        self.snames = [os.path.join(self.curr_scan_path, name) for name in snames]

        return self.snames

    def resave_image_volumes(self, X):
        """
        Re-saves image volumes into the processing folder for ease of access
        Args:
            X (4D numpy array): image volume (dims: Z, X, Y, num_vols)

        Returns:

        """

        # Get image dimensions
        sz = X.shape

        # Save image volumes
        for i in range(sz[-1]):

            tmp = X[:, :, :, i].squeeze().swapaxes(0, 1)
            nib.save(nib.Nifti1Image(tmp, np.eye(4)), self.snames[i])

    def save_dicom_header(self, dicom_fname):

        # Collect Dicom header
        dcm = dcmread(dicom_fname)

        # Set up output file
        sname = os.path.join(self.curr_scan_path, 'dicom_header.txt')

        # Write output file
        with open(sname, 'w') as f:
            f.write(dcm.__str__())

    @staticmethod
    def clear_working_directory():
        """
        Clears all files from the working directory.
        Returns:

        """

        # Get working directory
        working_path = os.path.join(os.getcwd(), 'Working')

        # Get a list if items in the working directory
        working_files = glob(os.path.join(working_path, '*.*'))

        # Delete working files
        for file in working_files:
            os.remove(file)


class ProcessAnimal:
    def __init__(self, snames):

        self.T1_file = snames[0]
        self.T1c_file = snames[1]
        self.T2_file = snames[2]
        self.cur_scan_path = os.path.split(snames[0])[0]
        self.mask_file = ''
        self.radiomic_files = []
        self.radiomics_sfile = []
        self.dilate = 25

        # init_dilation_class(self.dilate)

    def bias_correct(self):

        # Set up correction parameters

        # Get weighted image
        weight_im = self.T1_file

        # Get the T2 weighted image (the only one that needs bias correction)
        T2 = self.T2_file

        # Set up output image
        T2_out = os.path.join(self.cur_scan_path, 'T2_cor.nii.gz')

        # Set up the bias corrector command
        cmd = 'N4BiasFieldCorrection ' \
              '--bspline-fitting [ 1x1x1, 3 ] ' \
              '-d 3 ' \
              '--input-image "%s" ' \
              '--convergence [ 100x100x100, 0.005 ] ' \
              '--output "%s" ' \
              '--shrink-factor 4 ' \
              '--weight-image "%s" ' \
              '--histogram-sharpening [0.3, 0.01, 200]' % (T2, T2_out, weight_im)

        subprocess.Popen(cmd, shell=True).wait()

        self.T2_file = T2_out

    def segment_tumor(self, model_path):

        # Load threshold from metrics file
        mfile = os.path.join(model_path, 'metrics.txt')
        with open(mfile, 'r') as f:
            dat = f.readlines()

        # Find last threshold calculated from the training set
        threshold = 0.5
        for z in range(-1, -12, -1):
            if 'Best threshold' in dat[z]:
                threshold = [i for i in dat[z] if i.isdigit() or i == '.']
                threshold = float(''.join(threshold))
                break

        # Run segmentation
        t2, y_pred = seg_from_model(model_path=model_path,
                                    im_paths=[self.T1_file, self.T1c_file, self.T2_file],
                                    threshold=threshold)

        # Save the new segmentation
        self.mask_file = os.path.join(self.cur_scan_path, 'tumor_seg.nii.gz')
        nib.save(nib.Nifti1Image(y_pred, np.eye(4)), self.mask_file)

        # Make segmentation images
        display_segmentations(t2, y_pred, self.cur_scan_path)

    def compute_radiomics(self, animal_id):

        # Set up values for multiple mask configurations
        sfiles = ['radiomic_features.csv',
                  'radiomic_features_bed.csv',
                  'radiomic_features_edge.csv']
        dilate = [0, self.dilate, self.dilate]
        diff_mask = [False, False, True]

        # Set up empty list for radiomics files
        self.radiomics_sfile = []

        for i in range(3):
            if i == 0:
                regen = True
            else:
                regen = False

            # Data paths
            base_path = self.cur_scan_path
            save_base_path = os.path.join(os.getcwd(), 'Working')

            # Append the current radiomics file
            self.radiomics_sfile.append(os.path.join(base_path, sfiles[i]))

            # Generate CSV file of images/masks and re-save images as 16 bit
            csv_file = gen_images_csv([self.T1_file, self.T1c_file, self.T2_file],
                                      mask_file=self.mask_file,
                                      save_base_path=save_base_path,
                                      dilate=dilate[i],
                                      ncontrasts=3,
                                      regen=regen,
                                      diff_mask=diff_mask[i],
                                      animal_id=animal_id)

            # Run radiomics
            run_radiomics(self.cur_scan_path, csv_file, self.radiomics_sfile[i])

    def sort_radiomics(self, animal_id, study_date, base_path, summary_file):

        # Define data base path
        base_path = os.path.join(base_path, 'Results')

        # Load study data
        df = load_study_data(summary_file)

        # Define dictionary keys
        kid_key = 'Kirsch lab iD'
        rtd_key = 'Date of irradiation 20Gy x'

        # Convert study date to Datetime object for comparison
        study_dt = datetime.strptime(study_date, '%Y%m%d')

        # Convert the animal ID to a number for comparison
        an_id = int(animal_id.strip('K').strip('^'))

        # Get dataset animals
        animals = df[kid_key].astype('int')

        # Set up flags
        classified = False
        control_flag = False
        post_rt = False

        # Determine if the animal appears in the control of PD1 groups
        if any(an_id == animals):

            # Animal exists in database
            classified = True

            # Get animal study group
            group = df['Group'][an_id == animals].to_list()
            group = group[0]

            if group == 'Control':
                control_flag = True

            if group == 'PD1':
                control_flag = False

        else:
            print('Unclassified')

        # Set up paths to radiomics files in the animal directory
        rad_files = [os.path.join(self.cur_scan_path, file) for file in
                     ['radiomic_features.csv',
                      'radiomic_features_bed.csv',
                      'radiomic_features_edge.csv']
                     ]

        # Determine if the imaging was performed pre or post RT
        if classified:

            # Get RT date (already in Datetime)
            rt_date = df[rtd_key][an_id == animals]

            # Compare the RT and study date
            if all(rt_date < study_dt):

                # The study happened post-RT
                post_rt = True


            # Pre-RT, all groups are the same
            if not post_rt:
                fname = os.path.join(base_path, 'Radiomics_preRT.txt')

            # Control group, post RT
            if control_flag and post_rt:
                fname = os.path.join(base_path, 'Radiomics_control_postRT.txt')

            # PD1 group, post RT
            if not control_flag and post_rt:
                fname = os.path.join(base_path, 'Radiomics_PD1_postRT.txt')

        else:
            # If the animal does not appear in the excel sheet
            fname = os.path.join(base_path, 'Unfiled_animals.txt')

        # Write log file
        with open(fname, 'a') as f:
            for file in rad_files:
                f.write('%s\n' % file)
