import numpy as np
import tensorflow as tf

# Ensure TensorFlow 1.x compatibility
if not hasattr(tf, 'compat'):
    raise ValueError("This code requires TensorFlow V1.15.5, but you have an incompatible version.")

# Disable eager execution which is default in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

# Importing Keras modules directly from TensorFlow
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model

# Load the trained model
model = load_model('animal_classification_model.h5')

# Preprocess the input image
def preprocess_input_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Use the specified image path
input_image_path = 'D:/RS/Blocks_17JULRGB_linear/block_0_0.tif'
img_array = preprocess_input_image(input_image_path)

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# You'll need to specify the class labels manually
class_labels = ['class_1', 'class_2', 'class_3']  # Replace with your actual class labels

# Display results
print(f"Predicted class: {class_labels[predicted_class[0]]}")
print(f"Confidence scores per class: {predictions[0]}")
