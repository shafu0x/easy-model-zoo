import gdown 
import zipfile
from pathlib import Path

tmp = Path('/tmp')
dest = Path.home()/'.data'

def download_weights(model_name,weights_id):
    if not dest.is_dir(): dest.mkdir()
    for f in dest.iterdir():
        if model_name.lower() in f.stem:
            print('Loading cached model file') 
            return str(dest/f)

    cached_zip = False
    for f in tmp.iterdir():
        if model_name.lower() == str(f.stem.lower()):
            cached_zip = True

    file_name = f'{model_name}.zip'
    if not cached_zip:
        url = f'https://drive.google.com/uc?id={weights_id}'
        gdown.download(url, str(tmp/file_name), quiet=False)

    with zipfile.ZipFile(tmp/file_name, 'r') as zip_ref:
        extracted = zip_ref.namelist()[0]
        zip_ref.extractall(dest)

    return str(dest/extracted)

if __name__ == '__main__':
    print(download_weights('efficient',weights_id))

