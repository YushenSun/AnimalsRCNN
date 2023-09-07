from PIL import Image
import numpy as np

# Load image
image_path = 'D:/RS/Blocks_17JULRGB_linear_small10/block1_20_38.tif'
img = Image.open(image_path)

# Convert image to numpy array
img_np = np.array(img)

# Repeat each pixel 16 times along both x and y axis
enlarged_np = img_np.repeat(16, axis=0).repeat(16, axis=1)

# Convert numpy array back to image
enlarged_img = Image.fromarray(enlarged_np)

# Save the enlarged image
enlarged_img.save('D:/RS/enlarged/block1_20_38.tif')
