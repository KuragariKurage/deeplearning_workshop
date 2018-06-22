import os
import fnmatch
from scipy.io import loadmat, savemat

## Make directory if that don't exist
# path_dir: directory path to make
def make_dir(path_dir):
    if os.path.isdir(path_dir) is False:
        os.makedirs(path_dir)

## This function extracts full path of all files matched pattern under the target directory.
# directory: Target directory name
# pattern: File name pattern. You can use regular expression.
def find_all_files(directory, pattern="*"):
    for root, dirs, files in os.walk(directory):
#        yield root
        for file in files:
            if fnmatch.fnmatch(file, pattern)==True:
                yield os.path.join(root, file)