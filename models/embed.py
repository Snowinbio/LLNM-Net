from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.attention import Attention
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.rr_len
        num_img_fea = config.img_fea_len

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)                       # image datatype, if use multi images, we can set multi images to a 224*224
        self.rr_embeddings = Linear(768, config.hidden_size)  
        self.img_fea_embeddings = Linear(1, config.hidden_size)  
        self.sex_embeddings = Linear(1, config.hidden_size)  
        self.age_embeddings = Linear(1, config.hidden_size)  
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+n_patches, config.hidden_size))
        self.pe_rr = nn.Parameter(torch.zeros(1, tk_lim, config.hidden_size))
        self.pe_img_fea = nn.Parameter(torch.zeros(1, num_img_fea, config.hidden_size))
        self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_rr = Dropout(config.transformer["dropout_rate"])
        self.dropout_img_fea = Dropout(config.transformer["dropout_rate"])
        self.dropout_sex = Dropout(config.transformer["dropout_rate"])
        self.dropout_age = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, rr, img_fea, sex, age):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)                                            # Set 16*16 --> 1*1 with CNN
        rr = self.rr_embeddings(rr)
        img_fea = self.img_fea_embeddings(img_fea)
        sex = self.sex_embeddings(sex)
        age = self.age_embeddings(age)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        rr_embeddings = rr + self.pe_rr
        img_fea_embeddings = img_fea + self.pe_img_fea
        sex_embeddings = sex + self.pe_sex
        age_embeddings = age + self.pe_age

        embeddings = self.dropout(embeddings)
        rr_embeddings = self.dropout_rr(rr_embeddings)
        img_fea_embeddings = self.dropout_img_fea(img_fea_embeddings)
        sex_embeddings = self.dropout_sex(sex_embeddings)
        age_embeddings = self.dropout_age(age_embeddings)
        return embeddings, rr_embeddings, img_fea_embeddings, sex_embeddings, age_embeddings
