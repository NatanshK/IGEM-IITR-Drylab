# iGEM IIT ROORKEE
  The primary goal of this project is to build an image classification system that can accurately detect red rot disease in sugarcane plants based on images.

## Table of Contents
- [Dataset](#dataset)
  
### DATASET
  We found our dataset on https://data.mendeley.com/datasets/9424skmnrk/1. It contains 518 images of sugarcane leaves infected with reddot and 522 images of healthy sugarcane leaves.
We augmented out images in two steps, first we quadrupled our dataset by applying three rotations on every image. Further we doubled our dataset by applying standard data augmentation techniques like flipping, randomized cropping, introducing gaussian noise etc with some probabilities to maintain randomness.
  #### DATA CLEANING
