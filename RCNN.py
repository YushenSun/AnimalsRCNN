import torchvision.transforms as transforms
from osgeo import gdal
import os
import gdal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Define the list of target class labels
classes = ['elephant', 'lion', 'giraffe', 'zebra', 'rhino', 'non-animal']

# Assign unique integer codes to each target class
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Define the path to your dataset and annotation file
data_dir = 'D:/RS/Blocks_20SEP'
annotation_file = 'D:/RS/ano/20SEP.csv'  # Replace with the actual annotation file path

# Define a transform to preprocess the input image (if needed)
transform = transforms.Compose([
    # Add your transformations here if needed
])


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
                boxes = []
                category = line[1]
                bbox = [float(coord) for coord in line[2:]]
                boxes.append({'category': category, 'bbox': bbox})
                annotations.append({'image': image_name, 'boxes': boxes})
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

        num_boxes = len(annotation['boxes'])
        target = {
            'boxes': torch.tensor([box['bbox'] for box in annotation['boxes']], dtype=torch.float32),
            'labels': torch.tensor([self.class_to_idx[box['category']] for box in annotation['boxes']]),
        }

        return image, target

    def __len__(self):
        return len(self.annotations)


# Instantiate the CustomDataset
custom_dataset = CustomDataset(data_dir, annotation_file, transform)

# Create a DataLoader for training
train_loader = DataLoader(
    dataset=custom_dataset,
    batch_size=1,  # Adjust batch size as needed
    num_workers=0,  # You can increase this value for faster data loading
    shuffle=True,
    collate_fn=lambda x: list(zip(*x))  # This collates the data into batches
)

# Iterate through the train_loader
for batch in train_loader:
    images, targets = batch  # Extract the images and targets from the batch tuple
    images = torch.stack(images)  # Convert list of images to a tensor
    # Forward pass, loss calculation, and backpropagation here
    # Implement your training loop logic
