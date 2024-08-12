# AnimalsRCNN - Animal Detection and Classification with Mask R-CNN

## Overview

This repository contains an implementation of the **Mask R-CNN** algorithm for detecting and classifying animals in images. Mask R-CNN is an advanced deep learning model that not only identifies objects within an image but also provides pixel-level segmentation, making it ideal for tasks that require precise object delineation.

The project aims to develop a robust system for wildlife monitoring and research, using deep learning to analyze images captured in natural habitats.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Object Detection**: Detects animals in images with bounding boxes.
- **Instance Segmentation**: Provides pixel-level masks for detected animals.
- **Classification**: Classifies detected animals into predefined categories.
- **Model Training**: Includes scripts to train the Mask R-CNN model on custom datasets.
- **Evaluation**: Tools to evaluate model performance on test datasets.

## Dataset

The model is trained on a custom dataset containing images of various animals. Each image is annotated with bounding boxes, segmentation masks, and class labels.

- **Data Format**: The dataset should be in COCO format or similar, with separate files for training and validation sets.
- **Classes**: The dataset includes multiple animal classes, such as elephants, lions, and zebras.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/YushenSun/AnimalsRCNN.git
cd AnimalsRCNN
```

Ensure you have Python and the required libraries installed:

```bash
pip install -r requirements.txt
```

The project requires the installation of additional dependencies, such as TensorFlow, Keras, and OpenCV. You can install them with:

```bash
pip install tensorflow keras opencv-python
```

## Usage

### Training the Model

To train the Mask R-CNN model on your dataset:

1. Place your dataset in the `datasets/` directory.
2. Update the configuration file with the appropriate dataset path and parameters.
3. Run the training script:

   ```bash
   python train.py --dataset=path_to_your_dataset --epochs=50
   ```

### Inference

To perform inference on new images:

1. Place the images in the `images/` directory.
2. Run the inference script:

   ```bash
   python inference.py --image=path_to_image
   ```

The script will output the image with detected animals, along with their bounding boxes and masks.

### Evaluation

To evaluate the model performance on a test set:

```bash
python evaluate.py --dataset=path_to_test_dataset
```

This script will output metrics such as mean Average Precision (mAP) and segmentation accuracy.

## Results

The model achieves high accuracy in detecting and classifying animals across various test images. Below are some example results:

- **Elephant Detection**: 95% accuracy with clear segmentation.
- **Lion Detection**: 92% accuracy with well-defined bounding boxes.
- **Zebra Detection**: 94% accuracy with accurate masks.

Sample images from the test set, along with their corresponding detection and segmentation results, can be found in the `results/` directory.

## Technologies

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Instance Segmentation**: Mask R-CNN
- **Data Handling**: COCO format, JSON
- **Visualization**: Matplotlib, Seaborn

## Contributing

Contributions are welcome! If you want to improve the model, add new features, or fix bugs, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, suggestions, or feedback, feel free to contact me:

- **Yushen Sun**
- [LinkedIn](https://www.linkedin.com/in/syushen/)
- [Email](mailto:sun.yushen@gmail.com)
