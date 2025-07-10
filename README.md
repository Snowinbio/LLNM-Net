# LLNM-Net
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
## Explainable Multimodal Deep Learning for Predicting Thyroid Cancer Lateral Lymph Node Metastasis Using Ultrasound Imaging

## Train & Test
LLNM_Net.py contains the main code for training and testing the model. The models folder includes the components of the model and the code for model construction, with modeling_LLNM_Net.py being the model architecture, and config recording some model parameters. The detailed code that follows will be gradually open-sourced.

## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```
## Result
We have provided a thyroid LLNM (lateral lymph node metastasis) prediction 3d model based on locational information. Please see the webpage: index.html. url: https://snowinbio.github.io/LLNM-Net/

We visualized the focus points of LLNM-Net on the basis of GradCAM++ for location information, registered a large amount of data, and statistically analyzed the model prediction heatmaps based on location information. We then established a model of metastasis probability distribution in three-dimensional space.

## Data
All data supporting the research findings in this study are available in the article and its supplementary information files. The minimum dataset required to interpret, verify, and extend the study results on patients with lateral lymph node metastasis has been deposited in Hugging Face with the access link: https://huggingface.co/datasets/Snowinbio/LLNM_Multimodal_dataset. This includes: 1) Pre-processed cropped imaging data (ultrasound images with anonymized metadata). 2) Corresponding ultrasound imaging reports for patients, including professional descriptions of the images by physicians, characteristics of nodules, etc. 3) Clinical characteristic information of patients, including age and sex. The source data file containing detailed LLNM-Net outputs and key evaluation metrics can also be obtained via the following link: https://github.com/Snowinbio/LLNM-Net/blob/main/Source%20Data.xlsx. Due to ethical restrictions and patient confidentiality agreements, the full dataset (such as raw imaging data, detailed imaging reports, and patient clinical records) cannot be publicly available. This is because even after de-identification, detailed patient clinical records and high-resolution imaging data may still pose a risk of re-identification due to the unique characteristics of thyroid cancer cases. Researchers wishing to access additional data for non-commercial academic purposes may submit a formal application to the corresponding author. Applications will be reviewed by the institutional ethics committee and data custodians. The applicable conditions are as follows: 1) Purpose: The data may only be used for research purposes consistent with the original study objectives. 2) Access restrictions: Requestors must sign a data use agreement prohibiting re-identification or redistribution. 3) Data retention: Approved data will be available for 2 years from the date of publication. The data for each figure in this study are included in the "Source Data" section, with the file name Source Data.xlsx. This file can also be downloaded from the following link: https://github.com/Snowinbio/LLNM-Net/blob/main/Source%20Data.xlsx. Source data are provided in this article.

## Code
The primary code for the project is accessible at: https://github.com/Snowinbio/LLNM-Net.git. Installation instructions are provided in the repository. We have provided a permanent reference52 for the specific code version used in this study. The code is released under the Apache License 2.0, which permits free use, modification, and redistribution under its terms. The model implementation is built upon multiple publicly available open-source projects. We have retained all original license information and copyright notices in the corresponding source files. Specifically, we acknowledge the following contributions: IRENE (Apache 2.0): https://github.com/RL4M/IRENE, YOLOv8 by Ultralytics (AGPL-3.0): https://huggingface.co/Ultralytics/YOLOv8.

## Reference
All references are listed in the article.

## Licence
The code is distributed under the Apache License 2.0. It can be used for non-commercial purposes only after the publication of the article. For any commercial use, please contact the author for permission.
