import numpy as np
from PIL import Image

from src.efficientdet.efficientdetmodel import EfficientDetModel
from src.bisenet.bisenet import BisenetModel 
from src.yolact.run import YOLACTModel

# Google Drive ids for downloading the weights
EFFICIENTDET_D0='1g1SlGsR0ZQlWlW45S9JIpMPLrmwg1zvV'
BISENET='1jERXjLfxRNlFBt9UiYWZzxY4a7t-83_8'

class ModelRunner:
    def __init__(self, model_name, weights, device='CPU'):
        'Encapsulates a model'
        self.model = self.init_model(model_name, weights, device)

    def init_model(self, model_name, weights, device):
        if model_name == 'Bisenet': model = BisenetModel(model_name, weights, device)
        if model_name == 'YOLACT': model = YOLACTModel(model_name, weights, device)
        # There are 8 multiple EfficientDet variants
        if 'EfficientDet' in model_name: model = EfficientDetModel(model_name, weights, device)

        if model != None: print(f'{model_name} was initialized correctly.')
        else            : raise Exception(f'Model with name {model_name} could not be found!')

        return model

    def run(self, img):
        '`ìmage` can be the image file as string or the image as an array'
        return self.model.run(img)

    def calc_inf_time(self, n=10, sz=(850,650)):
        self.model._calc_inf_time(n, sz)

if __name__ == '__main__':
    img_path = '/home/sharif/Downloads/pp_gesicht.jpg'

    """
    # EfficientDet
    weights = '/home/sharif/Downloads/efficientdet-d1.pth'
    model_runner = ModelRunner('EfficientDet-d1', weights, 'GPU')
    #o = model_runner.run(img_path)
    #print(o)
    model_runner.calc_inf_time(10)
    """

    """
    # BiseNet
    weights = '/home/sharif/Desktop/BiSeNet/res/model_final.pth' 
    model_runner = ModelRunner('Bisenet', weights, 'GPU')
    model_runner.calc_inf_time(10)
    """

    # YOLACT
    weights = '/home/sharif/Downloads/yolact_resnet50_54_800000.pth' 
    model_runner = ModelRunner('YOLACT', weights, 'CPU')
    #o = model.run(img_path)
    #print(o)
    model_runner.calc_inf_time(10)
