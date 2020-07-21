import numpy as np
from PIL import Image

from .efficientdet import EfficientDetModel
from .bisenet.bisenet import BisenetModel 
from .yolact.run import YOLACTModel

from .download import download_weights

# Google Drive ids for downloading the weights
model_ids = {
    'EFFICIENTDET_D0':'1g1SlGsR0ZQlWlW45S9JIpMPLrmwg1zvV',
    'EFFICIENTDET_D1':'1rWuYVoe22NpUUfTlqAkxCQ69vyn0f_OQ',
    'EFFICIENTDET_D2':'1nqjApuGV-ejcqx8MjZJThRIo_8LMRTlc',
    'EFFICIENTDET_D3':'1eGJTEdxy40xTEJNUrTBYlUTJz-xZJGX1',
    'EFFICIENTDET_D4':'1YjxVzf7dOKdTiVMFgWmrTaqVtiwLHmkf',
    'EFFICIENTDET_D5':'1vYlC0X1d_AvlFoncYFLkMGWNU_UeoM6p',
    'EFFICIENTDET_D6':'127iopie3JDTPVv3d75XOzCYxo7nAQfYQ',
    'EFFICIENTDET_D7':'1j8cYgfylD4PTsyh8IC3qtwpDedx7eWPk',
    'BISENET':'1-xn4CkE33sq5ZuKh8IJAv79743ImHh7l',
    'YOLACT-RESNET50':'1PN_cBJRkyJzxdmlNU3wDIQlivwELsY0_',
}

class ModelRunner:
    def __init__(self, model_name, device='GPU'):
        self.model = self.init_model(model_name, device)

    def init_model(self, model_name, device):
        if model_name == 'Bisenet': model = BisenetModel(model_name, model_ids['BISENET'], device)
        if model_name == 'YOLACT-Resnet50': model = YOLACTModel(model_name, model_ids['YOLACT-RESNET50'], device)
        # There are 8 different EfficientDet variants
        if 'EfficientDet' in model_name:
            for i in range(8): 
                if str(i) in model_name: 
                    coeff = i; break
            model = EfficientDetModel(model_name, model_ids[f'EFFICIENTDET_D{i}'], device, coeff)

        if model != None: print(f'{model_name} was initialized correctly.')
        else            : raise Exception(f'Model with name {model_name} could not be found!')

        return model

    def run(self, img):
        '`ìmage` can be the image file as string or the image as an array'
        return self.model.run(img)

    def calc_inf_time(self, n=10, sz=(850,650)):
        self.model._calc_inf_time(n, sz)

    def visualize(self, image, pred):
        return self.model.visualize(image, pred)

if __name__ == '__main__':
    img_path = '/home/sharif/Documents/easy-model-zoo/tests/test.png'

    device = 'GPU'

    # EfficientDet
    model_runner = ModelRunner('EfficientDet-d0', device)
    #pred = model_runner.run(img_path)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d1', device)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d2', device)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d3', device)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d4', device)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d5', device)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d6', device)
    model_runner.calc_inf_time(10)
    model_runner = ModelRunner('EfficientDet-d7', device)
    model_runner.calc_inf_time(10)
    

    # BiseNet
    model_runner = ModelRunner('Bisenet', device)
    model_runner.calc_inf_time(10)
    #pred = model_runner.run(img_path)
    #model_runner.visualize(img_path, pred)
    #model_runner.calc_inf_time(10)
    
    # YOLACT 
    model_runner = ModelRunner('YOLACT-Resnet50', device)
    model_runner.calc_inf_time(10)
    #pred = model_runner.run(img_path)
    #print(pred)

    vis = model_runner.visualize(img_path, pred)
