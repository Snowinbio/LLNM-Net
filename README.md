# LLNM-Net
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
## Explainable Multimodal Deep Learning Method for Predicting Thyroid Cancer Lateral Lymph Node Metastasis Using Ultrasound Imaging

## Code
LLNM_Net.py contains the main code for training and testing the model. The models folder includes the components of the model and the code for model construction, with modeling_LLNM_Net.py being the model architecture, and config recording some model parameters. The detailed code that follows will be gradually open-sourced.

## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```

## Data
The data utilized in this study are subject to privacy restrictions. However, upon a reasonable request to the corresponding author, the data can be anonymized and made accessible for further analysis. We will provide sample data for readers to use.

## Result
We have provided a thyroid LLNM (lateral lymph node metastasis) prediction 3d model based on locational information. Please see the webpage: index.html. url: https://snowinbio.github.io/LLNM-Net/

We visualized the focus points of LLNM-Net on the basis of GradCAM++ for location information, registered a large amount of data, and statistically analyzed the model prediction heatmaps based on location information. We then established a model of metastasis probability distribution in three-dimensional space.

## Reference
All references are listed in the article.

## Licence
The code is distributed under the Apache License 2.0. It can be used for non-commercial purposes only after the publication of the article. For any commercial use, please contact the author for permission.
