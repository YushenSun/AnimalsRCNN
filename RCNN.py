# Define the list of target class labels
classes = ['elephant', 'lion', 'giraffe', 'zebra', 'rhino', 'non-animal']

# Assign unique integer codes to each target class
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# You can define more target classes and codes as needed

# Print the class labels and their corresponding integer codes
for cls, idx in class_to_idx.items():
    print(f'{cls}: {idx}')

import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None):
        """
        Initialize the CustomDataset.

        Args:
            data_dir (str): Path to the directory containing image data.
            annotation_file (str): Path to the annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = 'D:/RS'
        self.annotation_file = 'D:/RS/20SEP.tif'
        self.transform = transform
        self.class_to_idx = {'elephant': 0, 'lion': 1, 'giraffe': 2, 'zebra': 3, 'rhino': 4}
        self.annotations = self.load_annotations()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Gets a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): The image data as a tensor.
            target (dict): Dictionary containing target information including boxes and labels.
        """
        img_path = os.path.join(self.data_dir, self.annotations[idx]['image'])
        image = Image.open(img_path).convert('RGB')

        # Extract bounding box annotations and labels
        boxes = self.annotations[idx]['boxes']
        labels = [self.class_to_idx[annot['category']] for annot in boxes]
        boxes = torch.FloatTensor([annot['bbox'] for annot in boxes])

        # Construct target dictionary
        target = {'boxes': boxes, 'labels': labels}

        # Apply data transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, target

    def load_annotations(self):
        """
        Load annotations from the specified annotation file.

        Returns:
            annotations (list): List of dictionaries containing image filename and annotations.
        """
        annotations = []
        with open(self.annotation_file, 'rb') as f:
            lines = f.readlines()
        for line in lines:
            try:
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                parts = decoded_line.split(',')
                image_name = parts[0]
                boxes = []
                for box_str in parts[1:]:
                    box = box_str.split()
                    category = box[0]
                    bbox = [float(coord) for coord in box[1:]]
                    boxes.append({'category': category, 'bbox': bbox})
                annotations.append({'image': image_name, 'boxes': boxes})
            except Exception as e:
                print("Error processing line:", line)
                print("Error message:", e)
        return annotations


import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# Define your custom dataset class (assuming you have implemented it)
# from custom_dataset import CustomDataset

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define hyperparameters
num_classes = 2  # Number of classes (including background)
batch_size = 2
learning_rate = 0.001
num_epochs = 10

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
# Replace the classifier with a new one, customized for your number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define the data loaders for training and validation
train_dataset = CustomDataset(data_dir='D:/RS',
                              annotation_file='20SEP.tif')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)

# Move the model and data loaders to the appropriate device
model.to(device)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
