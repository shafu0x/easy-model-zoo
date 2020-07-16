# Easy-Model-Zoo

We don't discriminate against PyTorch, Tensorflow or anyone else ;) This is a home for evey **easy** runnable model.

Are you also frustrated by the installation process of different models? You are tired of Docker and C extensions failing while compiling. You just want to try out a new model? [I agree!](https://towardsdatascience.com/running-deep-learning-models-is-complicated-and-here-is-why-35a4e325486c) You just found the right place!

The only **requirement** of theses models is that they are pip installable.

You don't have a fancy GPU? Don't worry just run it on the CPU...

PRs are always welcome!

# Installation

Simply run:

`pip3 install -r requirements.txt`

# Pre-trained Models

**NOTE:** You do NOT have to download the weights file yourself. The `ModelRunner` will do that for you. The links are just for convenience.

**Benchmarks**: 
- All benchmarks include pre- and postprocessing.
- GPU used: **GeForce GTX 1070**
- CPU used: **Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz**

## Object Detection

For a full comparison with other Object Detection models see [here](https://paperswithcode.com/sota/object-detection-on-coco).

| Model Name | Speed in ms | Speed in FPS | COCO AP | Original Repo | Paper | Weights |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
EfficientDet-d0 | 55 | 18 | 33.8% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) 
EfficientDet-d1 | 66 | 15 | 39.6% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth)
EfficientDet-d2 | 87 | 11.5 | 43.0% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth)
EfficientDet-d3 | 142 | 7 | 45.8% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth)
EfficientDet-d4 | 250 | 4 | 49.4% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth)|
EfficientDet-d5 | 500 | 2 | 50.7% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth)
EfficientDet-d6 | 666 | 1.5 | 51.7% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070)| [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth)
EfficientDet-d7 | 763 | 1.31 | 53.7% | [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | [arxiv](https://arxiv.org/abs/1911.09070) | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth)

## Semantic Segmentation

| Model Name | MS (GPU) | FPS (GPU) | MS (CPU) | FPS (CPU)| Cityscapes MIOU  | Original Repo | Paper | Weights |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
Bisenet | 37 | 50  | 613 | 1.63 | 74.7%  | [here](https://github.com/CoinCheung/BiSeNet) | [arxiv](https://arxiv.org/abs/1808.00897)| [bisenet.pth](https://github.com/SharifElfouly/BiSeNet/blob/master/res/model_final.pth)

## Instance Segmentation

| Model Name | MS (GPU) | FPS (GPU) | MS (CPU) | FPS (CPU)| Cityscapes MIOU  | Original Repo | Paper | Weights |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
YOLACT-Resnet50 | 69 |14 | 1397 |0.72 | 28.2%  |[here](https://github.com/dbolya/yolact) | [arxiv](https://arxiv.org/abs/1904.02689)| [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)

# Getting Started

```
from emz import Model

IMG_F = 'Full path to your image'

# Initialize the model object with a model name.
# Note: The model automatically figures out what task it will be used for.
# You can initialize it on the GPU or on the CPU with the device attribute.
# If it can't find a GPU it will use the CPU.
# NOTE: You don't need to download the weights yourself. The script will do it for you.
model = Model('YOLACT', device='GPU')

# You can either run the model on an image path or on a numpy array.
pred = model.run(IMG_F)

# Visualize the results.
model.visualize(pred)
```

# How to add a new Model?

Adding a new model is easy. Simply create a new directory inside src with the name of your model. Define a new Model class that inherits from `src/model.py`. For an example look at `src/bisenet/bisenet.py`.

Just remember, it has to be pip installable.

# License
Feel free to do what you [want](https://github.com/SharifElfouly/pretrained-model-zoo/blob/master/LICENSE)! Just don't blame me if it doesn't work ;)
