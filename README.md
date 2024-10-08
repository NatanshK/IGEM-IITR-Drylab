# iGEM IIT ROORKEE
  The primary goal of this project is to build an image classification system that can accurately detect red rot disease in sugarcane plants based on images.

## Table of Contents
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Model Training](#model-training)
  - [CNN + Random Forest](#cnn--random-forest)
  - [Vision Transformers](#vision-transformers)
  - [VGG16](#VGG16)
  - [EfficientNet](#efficientnet)
  - [ResNet](#resnet)
- [Results](#results)

### Dataset

The dataset used for this project is available [here](https://data.mendeley.com/datasets/9424skmnrk/1). It consists of:
- **518 images** of sugarcane leaves infected with *redrot disease*.
- **522 images** of healthy sugarcane leaves.

### To enhance the diversity of the dataset and improve model generalization, we employed a two-step data augmentation process:
#### 1. Image Rotation
Each image in the dataset was subjected to three rotations, thereby quadrupling the size of the dataset.
#### 2. Further Augmentation
The dataset was further expanded by applying additional augmentation techniques like RandomResizedCrop, sharpening, introducing gaussian noise, flipping etc. with probabilistic randomness, which resulted in doubling the dataset size.
<table>
  <tr>
    <td>
      <h3>Healthy Leaf Example:</h3>
      <img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/Original_Dataset/Healthy/healthy%20(360).jpeg?raw=true" alt="Healthy Image" style="height: 500px;"/>
    </td>
    <td>
      <h3>Redrot Infected Example:</h3>
      <img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/Original_Dataset/RedRot/redrot%20(225).jpeg?raw=true" alt="Red Dot Infected Image" style="height: 500px;"/>
    </td>
  </tr>
</table>

### DATA CLEANING
The image cleaning process involves converting the images to grayscale and applying a binary threshold to create a binary representation that highlights the relevant features. Subsequently, the algorithm identifies the bounding box around the detected features, enabling the cropping of the image to eliminate extraneous background elements and focus on the region of interest.

<h3>Image Cleaning Code:</h3>
<pre>
<code>
def find_bounding_box(image, threshold=10):
    height, width, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    rows = np.any(binary_image, axis=1)
    cols = np.any(binary_image, axis=0)
    if np.any(rows) and np.any(cols):
        min_y = np.argmax(rows)
        max_y = len(rows) - 1 - np.argmax(rows[::-1])
        min_x = np.argmax(cols)
        max_x = len(cols) - 1 - np.argmax(cols[::-1])
        cropped_image = image[min_y:max_y+1, min_x:max_x+1]
    else:
        cropped_image = image
    return cropped_image
</code>
</pre>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 50%; vertical-align: top;">
      <h3>Original Image:</h3>
      <img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/DATASET/REDDOTOA/aug_redrot%20(1)_270.jpeg?raw=true" alt="Original Image" style="height: 500px; width: auto;"/>
    </td>
    <td style="width: 50%; vertical-align: top;">
      <h3>Cleaned Image:</h3>
      <img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/DATASET_NEW/REDROT/aug_redrot%20(1)_270.jpeg?raw=true" alt="Cleaned Image" style="height: 500px; width: auto;"/>
    </td>
  </tr>
</table>

### MODEL TRAINING
#### CNN + RANDOM FOREST
A Random Forest classifier was trained on features extracted from a fine-tuned ResNet-18 model, specifically to detect red rot disease in sugarcane leaves. This approach leverages the deep learning model's ability to generate informative feature representations for enhanced classification accuracy.

**Training Code**:
```python
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

cnn_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
cnn_model.fc = nn.Identity()
cnn_model = cnn_model.to(device)
cnn_model.eval()

def extract_features(dataloader, model):
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.vstack(features_list), np.hstack(labels_list)

rf_classifier = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt'] 
}
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
```
#### Vision Transformers
The Vision Transformer (ViT) model with a base configuration (ViT-base-patch16-224) was fine-tuned on the dataset, leveraging pretrained weights to enhance performance in detecting red rot disease in sugarcane leaves.

**Training Code**:
```python
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = timm.create_model('vit_base_patch16_224', pretrained=True)
num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
```
#### VGG16
VGG16 was fine-tuned on the dataset, leveraging pre-trained ImageNet weights.
**Training Code**:
```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
```
#### EFFICIENTNET
EfficientNet_b0 was trained on the augmented and cleaned dataset.

**Training Code**:
```python
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
```
#### RESNET
ResNet-152 was fine-tuned on the dataset, leveraging pre-trained ImageNet weights.

**Training Code**:
```python
import torch
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
```

### RESULTS
```markdown
Below is the comparison of accuracy for the four models:

| Model               | Accuracy |
|---------------------|----------|
| Hybrid CNN + RF     | 96.15%   |
| Vision Transformers | 96.51%   |
| VGG16               | 99.16%   |
| EfficientNet_b0     | 99.64%   |
| ResNet-152          | 99.88%   |
