link = 'https://drive.google.com/uc?id=1g1SlGsR0ZQlWlW45S9JIpMPLrmwg1zvV'

import gdown 
gdown.download(link, 'efficientdet-d0.zip' , quiet=False)

def unzip(): pass

import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
