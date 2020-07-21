import sys
sys.path.append("..")

print(sys.path)

from ..easy_model_zoo import ModelRunner

img_path = '/home/sharif/Documents/easy-model-zoo/tests/test.png'

device = 'GPU'

# EfficientDet
model_runner = ModelRunner('EfficientDet-d0', device)
model_runner.calc_inf_time(10)