import time
import numpy as np
from PIL import Image

class Model:
    def __init__(self, name, weights_f):
        self.name = name
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

    def visualize(self):
        pass

    def calc_inf_time(self, n, sz=(850,650)):
        'Run model `n` times on array of size `sz` and calc average inference time'
        img = np.random.randn(1, 3, sz[0], sz[1])
        times = []
        for i in range(n):
            s = time.time()
            self.model(img)
            e = time.time()
            times.append(e-s)
        print(f'Average inference time for array with size {sz} is {mean(times)} ms.')
