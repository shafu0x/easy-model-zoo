import gdown 
import zipfile
from pathlib import Path

weights_id = '1g1SlGsR0ZQlWlW45S9JIpMPLrmwg1zvV'
tmp = Path('/tmp')

def download(model_name, weights_id, dest):
    url = f'https://drive.google.com/uc?id={weights_id}'
    file_name = f'{model_name}.zip'
    gdown.download(url, str(tmp/file_name), quiet=False)

    with zipfile.ZipFile(tmp/file_name, 'r') as zip_ref:
            zip_ref.extractall(dest)

if __name__ == '__main__':
    download('efficient',weights_id, Path.cwd())

