import numpy as np
from PIL import Image

from efficientdet.run import Model as EfficientDetModel
from bisenet.bisenet import Model as BisenetModel 

class Model:
    def __init__(self, model_name, weights, device='CPU'):
        self.device = device
        self.model = self.init_model(model_name, weights)

    def init_model(self, model_name, weights):
        if model_name == 'EfficientDet-d1': model = EfficientDetModel(weights)
        if model_name == 'Bisenet': model = BisenetModel(weights)
        if model_name == 'YOLACT': return None

        if model != None: print(f'{model_name} was initialized correctly.')
        else            : raise Exception(f'Model with name {model_name} could not be found!')

        return model

    @staticmethod
    def img2arr(img): 
        if isinstance(img, str): img = np.array(Image.open(img))
        return img

    def run(self, img):
        '`ìmage` can be the image file as string or the image as an array'
        img_arr = Model.img2arr(img)

        return self.model.run(img_arr)

if __name__ == '__main__':
    img_path = '/home/sharif/Downloads/pp_gesicht.jpg'

    """
    # EfficientDet
    weights = '/home/sharif/Desktop/pretrained-model-zoo/weights/efficientdet-d1.pth'
    model = Model('EfficientDet-d1', weights, 'GPU')
    o = model.run(img_path)
    print(o)
    """

    # BiseNet
    weights = '/home/sharif/Desktop/BiSeNet/res/model_final.pth' 
    model = Model('Bisenet', weights, 'GPU')
    o = model.run(img_path)
    print(o)

