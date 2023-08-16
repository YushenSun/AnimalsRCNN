import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image, ImageDraw

# Define the list of target class labels
classes = ['elephant', 'cluster', 'non-animal']

# Define the backbone model for Faster R-CNN
backbone = resnet_fpn_backbone('resnet50', pretrained=True)

# Define the anchor generator
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),) * 5,  # Make sure to match the number of feature maps
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Create the Faster R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=len(classes),  # Number of target classes
    rpn_anchor_generator=rpn_anchor_generator
)

# Load the trained model weights
model.load_state_dict(torch.load('D:/RS/models/trained_model1.pth'))
model.eval()

# Load and preprocess the new image
image = Image.open('D:/RS/Blocks_17JULRGB/block_1_0.tif')
image_tensor = F.to_tensor(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    predictions = model(image_tensor)

# Initialize counters for each class
class_counters = [0] * len(classes)

# Post-processing and visualization
# Assuming you want to draw bounding boxes on the image
draw = ImageDraw.Draw(image)
detection_threshold = 0.5  # Set your own detection threshold
for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > detection_threshold:
        class_counters[label] += 1  # Increment the counter for the detected class
        draw.rectangle(xy=box.tolist(), outline='red', width=3)
        draw.text((box[0], box[1]), f'Class: {classes[label]} - Score: {score:.2f}', fill='red')

# Display the counts of detected objects for each class
for i, cls in enumerate(classes):
    print(f'Number of {cls} objects detected: {class_counters[i]}')

image.show()  # Display the image with detections