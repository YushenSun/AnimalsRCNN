import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

# Path to the annotation file
annotation_file = 'D:/RS/ano/17JUL.csv'

# Path to the image used for training
image_path = 'D:/RS/Blocks_17JULRGB/block_0_0.tif'

# Read annotation data from CSV file
annotations = pd.read_csv(annotation_file)

# Load and display the image
image = Image.open(image_path)
plt.imshow(image, interpolation='bilinear')  # Set interpolation method to 'bilinear'

# Define colors for different annotation types
color_map = {'elephant': 'red', 'cluster': 'blue', 'non-animal': 'green'}

# Plot bounding boxes
for idx, row in annotations.iterrows():
    category = row['category']
    x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
    width = x_max - x_min
    height = y_max - y_min
    color = color_map.get(category, 'black')
    rect = Rectangle((x_min, y_min), width, height, fill=False, edgecolor=color, linewidth=2)
    plt.gca().add_patch(rect)

# Hide axis labels and ticks
plt.axis('off')


# Show the image with bounding boxes
plt.show()
