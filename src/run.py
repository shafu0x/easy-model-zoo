from bisenet import run

model_f = '/home/sharif/Desktop/BiSeNet/res/model_final.pth'
img_f = '/home/sharif/Desktop/BiSeNet/pic.jpg'

out = run(model_f, img_f)
print(out)