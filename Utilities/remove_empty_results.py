import os
from time import time
import shutil


def get_empty_folders(root_path, items_allowed=2):

    # Folders to delete
    del_folders = []

    # Get a list of subdirs
    walker = os.walk(root_path)

    for w in walker:

        # Separate base folder, subdirectories, and files
        base = w[0]
        subdirs = w[1]
        files = w[2]

        # Check length of files in folder
        file_len = len(subdirs) == 0 and len(files) <= items_allowed

        # Check modified date
        file_dat = within_modify_date(base)

        if file_len and not file_dat:

            del_folders.append(base)

    return del_folders


def within_modify_date(path, creation_safety=60):
    """
    Compares the creation date of a folder or file with the current time.
    Returns True if the folder/file has not been modified with the
    creation_safety period.
    Args:
        path (str): path to file or folder
        creation_safety (str): time in seconds, default 60 mins

    Returns:
        bool: True if the folder/file has been modified within the specified
            time
    """

    # The modification time
    folder_time = os.path.getmtime(path)

    # Compare with current time
    output = (time() - folder_time) < creation_safety

    return output


def delete_folders(folders):

    for folder in folders:

        # Remove folder
        shutil.rmtree(folder)

        print('Removed %s' % folder)


if __name__ == "__main__":

    root_path = 'E:\\MR Data\\ML_Results'

    # Get empty folder names
    folders = get_empty_folders(root_path, items_allowed=5)

    # Delete empty folders
    delete_folders(folders)
