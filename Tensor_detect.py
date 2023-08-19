import numpy as np
import tensorflow as tf
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('animal_classification_model.h5')

# Preprocess the input image
def preprocess_input_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Replace with the exact image path you want to test
input_image_path = 'D:/RS/Blocks_17JULRGB_linear/block_0_0.tif'
img_array = preprocess_input_image(input_image_path)

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Assuming the train_generator is still available from your previous code
# If not, you'll need to recreate it or manually specify the class labels
class_labels = list(train_generator.class_indices.keys())

# Display results
print(f"Predicted class: {class_labels[predicted_class[0]]}")
print(f"Confidence scores per class: {predictions[0]}")

