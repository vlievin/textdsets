import os
import shutil
import urllib.request
import zipfile


from .parser import tokenize_words

URL = 'http://mattmahoney.net/dc/text8.zip'
SPLITS_RATIOS = (0.9, 0.05, 0.05)
FILENAME = 'text8.zip'
DIRECTORY = 'text8/'


def read_and_download_text8_data(root: str = 'data/') -> str:
    directory = os.path.join(root, DIRECTORY)
    file_path = os.path.join(directory, FILENAME)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.isfile(file_path):
        print('Downloading text8 dataset...')

        with urllib.request.urlopen(URL) as response, open(file_path, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)

    rawdata = zipfile.ZipFile(file_path).read('text8').decode('utf-8')

    return rawdata
