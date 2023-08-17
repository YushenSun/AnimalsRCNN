import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F, transforms  # Import the transforms module
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image, ImageDraw

# Define the list of target class labels
classes = ['elephant', 'cluster', 'non-animal']

# Define the backbone model for Faster R-CNN
backbone = resnet_fpn_backbone('resnet50', pretrained=True)

print(backbone)

# Define the anchor generator
rpn_anchor_generator = AnchorGenerator(
    sizes=((4, 8, 16, 32, 64),) * 5,  # Make sure to match the number of feature maps
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Create the Faster R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=len(classes),  # Number of target classes
    rpn_anchor_generator=rpn_anchor_generator
)


# Define a function to visualize intermediate feature maps
def visualize_feature_maps(model, image_tensor, layer_name):
    """
    Visualize the intermediate feature maps of a specific layer in the model.

    Args:
        model (torch.nn.Module): The object detection model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        layer_name (str): Name of the layer whose feature maps you want to visualize.
    """
    # Register forward hook to capture feature maps
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output)

    layer = model.backbone._modules[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    # Perform inference
    with torch.no_grad():
        model(image_tensor)

    # Unregister the hook
    hook_handle.remove()

    # Visualize the feature maps
    num_feature_maps = len(feature_maps)
    for i, feature_map in enumerate(feature_maps):
        batch_size, num_channels, height, width = feature_map.shape
        feature_map = feature_map[0]  # Take the first image in the batch
        feature_map = feature_map.detach().cpu()

        # Normalize the feature map for visualization
        feature_map -= feature_map.min()
        feature_map /= feature_map.max()
        feature_map *= 255

        # Convert the feature map to an image
        feature_map_image = transforms.ToPILImage()(feature_map)

        # Display the feature map image
        feature_map_image.show(title=f'Layer: {layer_name}, Feature Map: {i + 1}/{num_feature_maps}')


# Load the trained model weights
model.load_state_dict(torch.load('D:/RS/models/trained_model_anchor4_echo5.pth'))
model.eval()

# Load and preprocess the new image
image = Image.open('D:/RS/Blocks_17JULRGB/block_0_0.tif')

# Apply the same normalization used during training
normalize = transforms.Normalize(mean=[128.2, 106.21, 101.5], std=[17.06, 14.41, 10.31])
image_tensor = F.to_tensor(image)
image_tensor = normalize(image_tensor).unsqueeze(0)

# Visualize feature maps of a specific layer (e.g., 'layer3')
# visualize_feature_maps(model, image_tensor, 'layer1')


# Perform inference
with torch.no_grad():
    predictions = model(image_tensor)



# Initialize counters for each class
class_counters = [0] * len(classes)

# Post-processing and visualization
# Assuming you want to draw bounding boxes on the image
draw = ImageDraw.Draw(image)
detection_threshold = 0.1  # Set your own detection threshold
# detection_threshold = 0.5  # Set your own detection threshold
for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > detection_threshold:
        class_counters[label] += 1  # Increment the counter for the detected class
        draw.rectangle(xy=box.tolist(), outline='red', width=3)
        draw.text((box[0], box[1]), f'Class: {classes[label]} - Score: {score:.2f}', fill='red')

# Display the counts of detected objects for each class
for i, cls in enumerate(classes):
    print(f'Number of {cls} objects detected: {class_counters[i]}')

image.show()  # Display the image with detections