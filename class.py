import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from PIL import Image

# Load labels from CSV
label_df = pd.read_csv('D:/RS/ano/count11.csv')

# Create a dictionary to store counts
label_dict = {(row['x'], row['y']): row['count'] for _, row in label_df.iterrows()}

# Load images and labels
image_dir = 'D:/RS/Blocks_17JULRGB_linear_small'
X = []  # To store image data
y = []  # To store labels

for x in range(64):
    for y_coord in range(64):
        filename = f'block_{x}_{y_coord}.tif'
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        X.append(img_array)

        # Get label from the dictionary or set to 0 if not present
        label = label_dict.get((x, y_coord), 0)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Normalize pixel values
X = X.astype('float32') / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Assuming the maximum count is 9; if it's higher, increase this number
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Save the model if needed
model.save('count_model.h5')
