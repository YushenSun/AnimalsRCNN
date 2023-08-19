import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define data paths
data_dir = 'D:/RS/Blocks_17JULRGB_linear'
annotation_file = 'D:/RS/ano/17JUL.csv'

# Load annotations from CSV file
annotations = pd.read_csv(annotation_file)

# Load images and labels
image_paths = [os.path.join(data_dir, image_name) for image_name in annotations['image_name']]  # Check the column name here
labels = annotations['category'].values  # Check the column name here

# Create a DataFrame for easier manipulation
data = pd.DataFrame({'image_path': image_paths, 'label': labels})

# Define data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,     # Normalize pixel values
    rotation_range=20,      # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2, # Random vertical shift
    horizontal_flip=True    # Random horizontal flip
)

# Split the data into training and validation sets
train_data = data.sample(frac=0.8, random_state=42)
valid_data = data.drop(train_data.index)

# Create data generators
batch_size = 32
train_generator = datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),  # Resize images to this size
    batch_size=batch_size,
    class_mode='sparse'      # For categorical labels
)

valid_generator = datagen.flow_from_dataframe(
    valid_data,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse'
)

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Change 3 to the number of classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
num_epochs = 10
steps_per_epoch = len(train_generator)
validation_steps = len(valid_generator)

model.fit(train_generator,
          epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=valid_generator,
          validation_steps=validation_steps)

# Save the trained model
model.save('animal_classification_model.h5')

# Evaluate the model on test data (if available)
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')
