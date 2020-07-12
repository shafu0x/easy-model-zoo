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