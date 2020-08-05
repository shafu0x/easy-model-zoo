from easy_model_zoo import ModelRunner

img_path = '/home/sharif/Documents/easy-model-zoo/tests/test.png'

device = 'GPU'

# EfficientDet
model_runner = ModelRunner('EfficientDet-d0', device)
model_runner.run(img_path)

model_runner = ModelRunner('Bisenet', device)
model_runner.run(img_path)

model_runner = ModelRunner('YOLACT-Resnet50', device)
model_runner.run(img_path)
