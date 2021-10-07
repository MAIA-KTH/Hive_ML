import re
import os



_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def get_filepath(path, pattern):
    """
    Path: path to the data folder
    Pattern: string pattern in filenames of interest
    Returns the full path of each file containing
    pattern in the filename.
    """
    data_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if pattern in file:
                file_path = os.path.join(root, file)
                data_list.append(file_path)
    data_list.sort(key=natural_sort_key)
    return data_list