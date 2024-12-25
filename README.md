# Multimodality-task method
## Overview

### Image Data
The original dataset is from [Kaggle The Chinese Mammography Database (CMMD) 2022](https://www.kaggle.com/datasets/tommyngx/cmmd2022), which includes the pathological classification label and mammography classification label. 
We revised this dataset, and it now has corresponding lesion masks for segmentation.


### Step 1 preprocess
Use ```preprocess.py``` to extract roi of breast for preprocess
### Step 2 Coordinates

```sample of caption.txt``` is sample of caption for coordinates encoding.

The coordinates of the centroid of the breast are determined by ```centroid.py```

### Step 3 Alignment

```caption.txt``` is sample of diagnostic reports for pretrained alignment task.

The training of alignment referred to [Simple CLIP](https://github.com/moein-shariatnia/OpenAI-CLIP?tab=readme-ov-file)

```alignment```folder contains hyperparameters and model selection.

Here, we used the ResNet 50 and Distilbert as image encoder and text encoder respectively.


### Step 4 Train

```multi_train.py```

please convert each path to your actual path

### Step 5 Predict for one image
```infer.py```
This file is for predicting each image and the output are segmentation maps, pathological classification, and mammography classification.

## Acknowledgement

Thanks to authors Shariatnia and M. Moein for their contributions in our alignment task of breast cancer information.
