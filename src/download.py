import gdown 
import zipfile
from pathlib import Path

weights_id = '1g1SlGsR0ZQlWlW45S9JIpMPLrmwg1zvV'
tmp = Path('/tmp')
dest = Path.home()/'.data'

def download_weights(model_name,weights_id):
    if not dest.is_dir(): dest.mkdir()

    url = f'https://drive.google.com/uc?id={weights_id}'
    file_name = f'{model_name}.zip'
    gdown.download(url, str(tmp/file_name), quiet=False)

    with zipfile.ZipFile(tmp/file_name, 'r') as zip_ref:
        extracted = zip_ref.namelist()[0]
        zip_ref.extractall(dest)

    return str(dest/extracted)

if __name__ == '__main__':
    print(download_weights('efficient',weights_id))

