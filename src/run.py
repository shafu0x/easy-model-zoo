import numpy as np
from PIL import Image

from src.efficientdet.efficientdetmodel import EfficientDetModel
from src.bisenet.bisenet import Model as BisenetModel 
from src.yolact.run import Model as YOLACTModel

class ModelRunner:
    def __init__(self, model_name, weights, device='CPU'):
        self.device = device
        self.model = self.init_model(model_name, weights)

    def init_model(self, model_name, weights):
        if model_name == 'EfficientDet-d1': model = EfficientDetModel('EfficientDet-d1', weights)
        if model_name == 'Bisenet': model = BisenetModel(weights)
        if model_name == 'YOLACT': model = YOLACTModel(weights)

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

    # EfficientDet
    weights = '/home/sharif/Downloads/efficientdet-d1.pth'
    model_runner = ModelRunner('EfficientDet-d1', weights, 'GPU')
    #o = model_runner.run(img_path)
    #print(o)
    model_runner.calc_inf_time(10)


    """
    # BiseNet
    weights = '/home/sharif/Desktop/BiSeNet/res/model_final.pth' 
    model = Model('Bisenet', weights, 'GPU')
    for _ in range(100):
        import time
        s = time.time()
        o = model.run(img_path)
        e = time.time()
        print(1/(e-s))
    """

    """
    # YOLACT
    weights = '/home/sharif/Downloads/yolact_resnet50_54_800000.pth' 
    model = Model('YOLACT', weights, 'GPU')
    o = model.run(img_path)
    print(o)
    """
