import time
import numpy as np
from PIL import Image

from src.download import download_weights

class Model:
    def __init__(self, name, weights_id, device='GPU'):
        self.name = name 
        self.weights_f = download_weights(name, weights_id)
        if device == 'GPU': self.use_cuda = True
        else              : self.use_cuda = False
        self.model = self._init_model()

    def _init_model(self): raise NotImplementedError('You will need to overwrite the `_init_model` method.')

    @staticmethod
    def img2arr(img): 
        if isinstance(img, str): img = np.array(Image.open(img).convert('RGB'))
        return img

    def _preprocess(self, image): 
        '`image` can be array or full path to image file.'
        pass

    def run(self, image): 
        'Run the model on `image`. `image` can be array or full path to image file.'
        raise NotImplementedError('You will need to overwrite the `run` method.')

    def visualize(self, pred):
        'Vis `pred` on the image'
        raise NotImplementedError('You will need to overwrite the `visualize` method.')

    def _calc_inf_time(self, n=100, sz=(850,600)):
        'Run model `n` times on array of size `sz` and calc average inference time'
        img = np.random.randn(1, sz[0], sz[1], 3)
        print(f'Calculating inference time for image with size {sz}:')

        # First inference is usually much slower. Therefore it is not used for the calculation.
        self.run(img)

        times = []
        for i in range(n):
            print(f'{i+1}/{n}')
            s = time.time()
            self.run(img)
            e = time.time()
            times.append(e-s)
        mean = sum(times)/len(times)
        print(f'Average inference time: {np.round(mean,3)} ms.')
        print(f'FPS: {np.round(1/mean,2)}')
