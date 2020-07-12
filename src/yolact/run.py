from .yolact import Yolact
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from .utils.augmentations import BaseTransform, FastBaseTransform, Resize
from .utils.functions import SavePath
import cv2

from .data import set_cfg

def evalimage(net:Yolact, img, save_path:str=None):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    
    frame = torch.from_numpy(img).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    
    preds = net(batch)
    return preds

class Model:
    def __init__(self, weights):
        cuda = True
        config = None

        if config is None:
            model_path = SavePath.from_str(weights)
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
            net.load_weights(weights)
            net.eval()
            print(' Done.')

            if cuda:
                net = net.cuda()
        self.model = net
    
    def run(self, img):
        return evalimage(self.model, img)
        

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