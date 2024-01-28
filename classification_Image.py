import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('sugarcane_MobileNetSmall_Custom.h5')

class_names = {0: 'Early Shoot Borer',
 1: 'Healthy',
 2: 'Mosaic',
 3: 'RedRot',
 4: 'Rust',
 5: 'Wolly Aphid'}

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_class(image_path, model, class_names):
    # Load the image
    img = load_img(image_path, target_size=(224, 224))

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Expand the dimensions of the numpy array
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class of the image
    class_prediction = model.predict(img_array)

    # Get the predicted class index
    class_index = np.argmax(class_prediction)

    class_name = class_names[class_index]
    confidence_score = class_prediction[0][class_index]
    # Return the predicted class name
    return class_name

user_input = input()
path = user_input[1:-1]
class_index = predict_class(path, model, class_names)
print(class_index)