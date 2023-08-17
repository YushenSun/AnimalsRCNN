import torchvision.transforms as transforms
from osgeo import gdal
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define the list of target class labels
classes = ['elephant', 'cluster', 'non-animal']

# Assign unique integer codes to each target class
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Define the path to your dataset and annotation file
data_dir = 'D:/RS/Blocks_17JULRGB'
annotation_file = 'D:/RS/ano/17JUL.csv'  # Replace with the actual annotation file path

# CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None):
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.annotations = self.load_annotations()

    def load_annotations(self):
        annotations = []
        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip the header line
                line = line.strip().split(',')
                image_name = line[0]
                category = line[1]
                bbox = [float(coord) for coord in line[2:]]
                annotations.append({'image': image_name, 'boxes': [bbox], 'category': [category]})
            return annotations

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_name = annotation['image']
        image_path = os.path.join(self.data_dir, image_name)

        # Open image using GDAL
        dataset = gdal.Open(image_path)
        image = dataset.ReadAsArray()

        # Convert to torch tensor and specify the data type
        image = torch.from_numpy(image.astype(np.float32))  # or np.float64

        target = {
            'boxes': torch.tensor(annotation['boxes'], dtype=torch.float32),
            'labels': torch.tensor([self.class_to_idx[cat] for cat in annotation['category']]),
        }

        return image, target

    def __len__(self):
        return len(self.annotations)

# Define a transform to preprocess the input image (if needed)
class ChannelwiseNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, image):
        num_channels = image.size(0)  # Get the number of channels in the image
        mean = self.mean[:num_channels]  # Use only the required mean values
        std = self.std[:num_channels]    # Use only the required std values
        image = (image - mean.view(num_channels, 1, 1)) / std.view(num_channels, 1, 1)
        return image


# Instantiate the CustomDataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    ChannelwiseNormalize(mean=[128.2, 106.21, 101.5],
                         std=[17.06, 14.41, 10.31]),
])


# Instantiate the CustomDataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
])

# Print the shape of mean and std
#print("Mean shape:", np.array([262]).shape)
#print("Std shape:", np.array([13]).shape)

custom_dataset = CustomDataset(data_dir, annotation_file, transform)


custom_dataset = CustomDataset(data_dir, annotation_file, transform)

# Create a DataLoader for training
train_loader = DataLoader(
    dataset=custom_dataset,
    batch_size=1,  # Adjust batch size as needed
    num_workers=0,  # You can increase this value for faster data loading
    shuffle=True,
    collate_fn=lambda x: list(zip(*x))  # This collates the data into batches
)

# ... (rest of the code for the model, training loop, etc.)
# ... (previous code)

# Define the object detection model
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Define the backbone model for Faster R-CNN
backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=True)

# Define the anchor generator
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),) * 5,  # 确保与 feature maps 数量匹配
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)


# Create the Faster R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=len(classes),  # Number of target classes
    rpn_anchor_generator=rpn_anchor_generator
)



# Training loop
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

'''
# Define optimizer and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Set the model in training mode
model.train()

# Number of training epochs
num_epochs = 10
'''


# Define optimizer and learning rate scheduler with modified parameters
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # Reduce LR more frequently

# Set the model in training mode
model.train()

# Number of training epochs for the trial run
num_epochs = 2  # Train for only a few epochs







# Define the device (GPU if available, otherwise CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

losses = []  # List to store loss values


# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch

        # Transfer images and targets to the device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Modify the dimensions of the target boxes to match the number of channels in images
        for t in targets:
            if 'boxes' in t:
                pass
        # Zero the gradients
        optimizer.zero_grad()

        # Print shapes of images and targets for debugging
        print(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}]")
        print("Images shape:", images[0].shape)  # Print shape of the first image in the batch
        print("Targets boxes shape:", targets[0]['boxes'].shape)  # Print shape of boxes in the first target
        print("Targets labels shape:", targets[0]['labels'].shape)  # Print shape of labels in the first target

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        total_loss = sum(loss for loss in loss_dict.values())

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Append the loss to the list
        losses.append(total_loss.item())

    # Update the learning rate
    lr_scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {total_loss.item()}")

# ... (continue with validation, evaluation, inference, and saving the model)

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Save the trained model
save_path = 'D:/RS/models'
model_name = 'trained_model1.pth'
torch.save(model.state_dict(), os.path.join(save_path, model_name))

