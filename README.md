# iGEM IIT ROORKEE
  The primary goal of this project is to build an image classification system that can accurately detect red rot disease in sugarcane plants based on images.

## Table of Contents
- [Dataset](#dataset)
  
### DATASET
  We found our dataset [here](https://data.mendeley.com/datasets/9424skmnrk/1). It contains 518 images of sugarcane leaves infected with reddot and 522 images of healthy sugarcane leaves.
We augmented out images in two steps, first we quadrupled our dataset by applying three rotations on every image. Further we doubled our dataset by applying standard data augmentation techniques like flipping, randomized cropping, introducing gaussian noise etc with some probabilities to maintain randomness.
#### Healthy Leaf Example: 
<img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/Original_Dataset/Healthy/healthy%20(360).jpeg?raw=true" alt="Healthy Image" height="500" style="margin-right: 200px;"/>
#### Reddot Infected Image:
<img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/Original_Dataset/RedRot/redrot%20(225).jpeg?raw=true" alt="Red Dot Infected Image" height="500"/>



  #### DATA CLEANING

## Table of Contents
- [Dataset](#dataset)
- [Image Augmentation and Cleaning](#image-augmentation-and-cleaning)
- [Model Training](#model-training)
  - [EfficientNet](#efficientnet)
  - [ResNet](#resnet)
  - [Hybrid CNN + Random Forest](#hybrid-cnn-random-forest)
- [Results](#results)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset was collected from various sources and consists of two classes: Healthy and Red Dot Infected sugarcane leaves.

- **Healthy Class**: 4000 images
- **Red Dot Class**: 4000 images

![Healthy Image](https://path_to_image.com/healthy_example.png) ![Red Dot Infected Image](https://path_to_image.com/reddot_example.png)

You can find the dataset [here](https://github.com/NatanshK/IGEM-IITR-Drylab).

## Image Augmentation and Cleaning

To improve the robustness of the model, we augmented and cleaned the images. Below is a comparison of an uncleaned vs cleaned image.

**Original Image**:
![Uncleaned Image](https://path_to_image.com/uncleaned.png)

**Cleaned Image**:
![Cleaned Image](https://path_to_image.com/cleaned.png)

### Code for Augmentation and Cleaning:
```python
import cv2
from albumentations import Compose, HorizontalFlip, Rotate

# Augmentation pipeline
augmentations = Compose([HorizontalFlip(p=0.5), Rotate(limit=45, p=0.5)])

def augment_image(image_path):
    image = cv2.imread(image_path)
    augmented = augmentations(image=image)
    return augmented["image"]

def clean_image(image):
    # Cleaning logic (e.g., noise removal, resizing)
    cleaned_image = cv2.GaussianBlur(image, (5, 5), 0)
    return cleaned_image

#### ResNet
```markdown
## ResNet
ResNet-152 was fine-tuned on the dataset, leveraging pre-trained ImageNet weights.

**Training Code**:
```python
import torch
from torchvision.models import resnet152

model = resnet152(pretrained=True)
model.fc = torch.nn.Linear(2048, 2)  # Adjusting for 2 classes

## How to Run

To replicate the project, follow these steps:

1. Clone the repo:
   ```bash
   git clone https://github.com/NatanshK/Sugarcare.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py --model efficientnet
   ```

4. Test the model:
   ```bash
   python test_model.py --model efficientnet
   ```

