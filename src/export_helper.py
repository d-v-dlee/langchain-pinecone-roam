import logging
import os

from glob import glob

# instantiate logger
logger = logging.getLogger('export_helper')
logger.setLevel(logging.INFO)

def find_n_latest_exports(files, n=1):
    # assumes the files are in the format: roam_export_MM-DD-YY.zip or MM-DD-YYYY.zip
    dates = [x.split('/')[-1].split('_')[-1].split('.')[0] for x in files]

    # Sort the dates in descending order
    sorted_dates = sorted(dates, reverse=True, key=lambda x: (int(x.split('-')[2]), int(x.split('-')[0]), int(x.split('-')[1])))

    # Take the first two elements of the sorted list
    top_two_dates = sorted_dates[:n]
    logger.info(f'The top {n} dates are: {top_two_dates}')

    # Find the indices of the top two dates in the original list
    indices = [dates.index(date) for date in top_two_dates]

    latest_zip_files = [files[i] for i in indices]
    return latest_zip_files

class RoamUnpacker():
    """
    the class accepts a path to a directory containing zips of Roam exports. The exports are in markdwon format.
    It will find the two latest exports and then unzip the files. It will then compare the differences between the two.
    """
    def __init__(self, path='../data/roam_exports/zip_files'):
        self._path = path
    
    def find_latest_export(self):
        """
        find the two latest exports
        """
        # load files
        if self._path[-1] != '/':
            self._path += '/'
        files = sorted(glob(self._path + "*.zip"))
        logger.info(f'Number of zip files: {len(files)}')

        if len(files) > 1:
            # first find the the two latest
            latest, second_latest = find_n_latest_exports(files, n=2)

            temp_latest_path, temp_second_latest_path = self._path + 'temp_latest', self._path + 'temp_second_latest'
            
            # create a temporary folder to unzip the files
            os.system(f'mkdir -p {temp_latest_path}')
            os.system(f'mkdir -p {temp_second_latest_path}')

            # unzip the files
            os.system(f'unzip {latest} -d {temp_latest_path}')
            os.system(f'unzip {second_latest} -d {temp_second_latest_path}')

            # find the differences between the two
            latest_files = set(sorted(glob(temp_latest_path + "/*.md")))
            second_latest_files = set(sorted(glob(temp_second_latest_path + "/*.md")))



        