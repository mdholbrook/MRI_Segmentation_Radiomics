import os
from time import time
import datetime
from Crawler.utilities import DataLoader, SaveResults, ProcessAnimal
from Crawler.update_radiomics_lists import update_radiomics_list


def run_crawler():

    tstart = time()

    # Set up data paths
    path = '/media/matt/Seagate Expansion Drive/b7TData_19/b7TData'
    path_regex = '*.dcm'
    log_file = os.path.join(path, 'Results/processing_log.json')
    summary_file = os.path.join(path, 'Results/Summary.xlsx')

    # Ensure results path exists
    out_path = os.path.join(path, 'Results')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Model
    # model_path = '/media/blkbeauty3/Matt/ML_Sarcoma_Results/2019_01_22_01-33-59_skip_all_contrasts_lr2e-4_400ep/'
    model_path = '/media/matt/Seagate Expansion Drive/MR Data/ML_Results/' \
                 '2019_11_09_14-12-47_cnn_model_3D_3lyr_do_relu_xentropy_skip'

    # Define data management classes
    loader = DataLoader(base_path=path, folder_regex=path_regex, log_file=log_file)
    saver = SaveResults(data_base_path=path, save_folder='Results', log_file=log_file)

    # Run crawler
    flag = True
    while flag:

        # Get working folder
        folder_exists = loader.get_folder()

        if folder_exists:  # or if the generator is not exhausted

            # Get animal name, study date, acquisition time, and protocol
            animal_id, study_date = loader.load_study_data()

            # Compare to animals already processed
            already_processed = loader.compare_with_log()

            if not already_processed:

                # Generate save paths
                snames = saver.gen_save_path(animal_id=animal_id, study_date=study_date)

                # Load data
                data = loader.load_dicom()

                # Save Nifti images to path
                saver.resave_image_volumes(X=data)
                saver.save_dicom_header(loader.names[0][0])

                # Set up processing functions for this animal
                process = ProcessAnimal(snames)

                # Make bias-corrected T2 images
                process.bias_correct()

                # Process data - Segmentation
                process.segment_tumor(model_path)

                # Process data - Compute radiomics
                radiomics_sfiles = process.compute_radiomics(animal_id)

                # Update report

                # Clear working directory
                saver.clear_working_directory()

                # Update log file
                saver.append_to_log(animal_id=animal_id, study_date=study_date)

                # Save the updated log file
                saver.save_log()

        else:

            flag = False

    # Update radiomics paths
    update_radiomics_list(out_path, summary_file)

    print('\tTotal time (HH:MM:SS): %s\n\n' % (str(datetime.timedelta(seconds=round(time() - tstart)))))


if __name__ == "__main__":

    run_crawler()
