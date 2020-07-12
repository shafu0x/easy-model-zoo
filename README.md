# Pretrained-Model-Zoo

Are you also frustrated by the installation process of different models? You are tired of Docker and C extensions failing while compiling. You just want to try out a new model to see how fast and accurate it is? [I agree!](https://towardsdatascience.com/running-deep-learning-models-is-complicated-and-here-is-why-35a4e325486c) You just found the right place!

The only **requirement** of theses models is that they are pip installable.

PRs are always welcome!

# Installation

Simply run:

`pip3 install -r requirements.txt`

# Pre-trained Models

## Object Detection

| Model Name | Speed (GTX 1070) in FPS | COCO AP | Weights | Original Repo | Paper |
| ----- | ----- | ----- | ----- | ----- | ----- |
EfficientDet-d1 | 23 | 39.6% | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth)| [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070) 

## Semantic Segmentation

| Model Name | Speed (GTX 1070) in FPS | Cityscapes MIOU | Weights | Original Repo | Paper |
| ----- | ----- | ----- | ----- | ----- | ----- |
Bisenet | 50 | 74.7% | [bisenet.pth](https://github.com/SharifElfouly/BiSeNet/blob/master/res/model_final.pth) | [here](https://github.com/CoinCheung/BiSeNet) | [arxiv](https://arxiv.org/abs/1808.00897)

## Instance Segmentation

| Model Name | Speed (GTX 1070) in FPS | COCO MAP | Weights | Original Repo | Paper |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
YOLACT (Resnet50-FPN) | 27 | 28.2% | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing) |[here](https://github.com/dbolya/yolact) | [arxiv](https://arxiv.org/abs/1904.02689)

# Getting Started

```
from ptz import Model

IMG_F = 'Full path to your image'

# Initialize the model object with a model name.
# Note: The model automatically figures out what task it will be used for.
# You can initialize it on the GPU or on the CPU with the device attribute.
# If it can't find a GPU it will use the CPU.
model = Model(model_name='YOLACT', device='GPU')

# You can either run the model on an image path or on a numpy array.
pred = model.run(IMG_F)

# Visualize the results.
model.visualize(pred)
```

# License
Feel free to do what you [want](https://github.com/SharifElfouly/pretrained-model-zoo/blob/master/LICENSE)!