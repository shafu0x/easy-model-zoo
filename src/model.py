import time
import numpy as np
from PIL import Image

class Model:
    def __init__(self, name, weights_f, device='GPU'):
        self.name = name 
        if device != 'GPU': self.use_cuda = False
        else              : self.use_cuda = True
        self.model = self._init_model(weights_f)

    def _init_model(self, weights_f): 
        'Initialize model with the full path to the weights file'
        pass

    @staticmethod
    def img2arr(img): 
        if isinstance(img, str): img = np.array(Image.open(img))
        return img

    def _preprocess(self, image): 
        '`image` can be array or full path to image file.'
        pass

    def run(self, image): 
        'Run the model on `image`. `image` can be array or full path to image file.'
        pass

    def visualize(self, pred):
        'Vis `pred` on the image'
        pass

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
