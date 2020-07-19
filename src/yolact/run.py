from .yolact import Yolact
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from .utils.augmentations import BaseTransform, FastBaseTransform, Resize
from .utils.functions import SavePath
import cv2

from .data import set_cfg

from ..model import Model


class YOLACTModel(Model):
    def __init__(self,name, weights_id, device='GPU'):
        super().__init__(name, weights_id, device)
    
    def _init_model(self):
        config = None

        if config is None:
            model_path = SavePath.from_str(self.weights_f)
            # TODO: Bad practice? Probably want to do a name lookup instead.
            config = model_path.model_name + '_config'
            print('Config not specified. Parsed %s from the file name.\n' % config)
            set_cfg(config)

        with torch.no_grad():

            if self.use_cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')     

            print('Loading model...', end='')
            net = Yolact()
            net.load_weights(self.weights_f)
            net.eval()
            print(' Done.')

            if self.use_cuda:
                net = net.cuda()
            return net
    
    def evalimage(self,net:Yolact, img, save_path:str=None):
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        
        img = Model.img2arr(img).reshape(1,3,1038,1188)
        print(img.shape)

        frame = torch.from_numpy(img).float()
        if self.use_cuda: frame = frame.cuda()
        #batch = FastBaseTransform()(frame.unsqueeze(0))

        preds = net(frame)
        return preds

    def run(self, img):
        return self.evalimage(self.model, img)

    def visualize(self, img, preds):
        from .eval import prep_display

        img = Model.img2arr(img).reshape(3,1038,1188)

        img = torch.Tensor(img)

        return prep_display(preds, img, 1038, 1188)
        

if __name__ == '__main__':
    trained_model = "weights/yolact_resnet50_54_800000.pth"
    image_f = '/home/sharif/Downloads/pic.jpg'
    cuda = True

    config = None

    if config is None:
        model_path = SavePath.from_str(trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % config)
        set_cfg(config)

    with torch.no_grad():

        if cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')     

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(trained_model)
        net.eval()
        print(' Done.')

        if cuda:
            net = net.cuda()

        evaluate(net, image_f)  
