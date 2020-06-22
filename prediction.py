import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Model saved with Keras model.save()
MODEL_PATH = 'vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()

img = image.load_img("img_path", target_size=(224, 224))

# Preprocessing the image
x = image.img_to_array(img)
# x = np.true_divide(x, 255)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
result = str(pred_class[0][0][1])               # Convert to string
print(result)