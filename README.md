# iGEM IIT ROORKEE
  The primary goal of this project is to build an image classification system that can accurately detect red rot disease in sugarcane plants based on images.

## Table of Contents
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- - [Model Training](#model-training)
  - [EfficientNet](#efficientnet)
  - [ResNet](#resnet)
  - [CNN + Random Forest](#cnn-random-forest) 
### DATASET
  We found our dataset [here](https://data.mendeley.com/datasets/9424skmnrk/1). It contains 518 images of sugarcane leaves infected with reddot and 522 images of healthy sugarcane leaves.
We augmented out images in two steps, first we quadrupled our dataset by applying three rotations on every image. Further we doubled our dataset by applying standard data augmentation techniques like flipping, randomized cropping, introducing gaussian noise etc with some probabilities to maintain randomness.
<table>
  <tr>
    <td>
      <h3>Healthy Leaf Example:</h3>
      <img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/Original_Dataset/Healthy/healthy%20(360).jpeg?raw=true" alt="Healthy Image" style="height: 500px;"/>
    </td>
    <td>
      <h3>Reddot Infected Example:</h3>
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
      <img src="https://github.com/NatanshK/IGEM-IITR-Drylab/blob/main/DATASET_NEW/REDDOT/aug_redrot%20(1)_270.jpeg?raw=true" alt="Cleaned Image" style="height: 500px; width: auto;"/>
    </td>
  </tr>
</table>

### MODEL TRAINING
#### EFFICIENTNET
EfficientNet was trained on the augmented and cleaned dataset.

**Training Code**:
```python
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


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


