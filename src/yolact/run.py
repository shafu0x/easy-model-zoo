from yolact import Yolact
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import SavePath
import cv2

from data import set_cfg


def evalimage(net:Yolact, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    import time
    for _ in range(100):
        s = time.time()
        preds = net(batch)
        e = time.time()
        print(1 / (e-s))

def evaluate(net:Yolact, image, train_mode=False):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    #cfg.mask_proto_debug = False

    evalimage(net, image)

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