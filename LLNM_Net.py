from __future__ import print_function, division 
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import pickle
import pandas as pd
from PIL import Image
import argparse
from apex import amp
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from models.modeling_LLNM_Net import LLNM_Net, CONFIGS
from tqdm import tqdm
import argparse
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from transformers import BertTokenizer, BertModel

tk_lim = 50  # report limit

disease_list = ['NonMeta', 'Latral']

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading LLNM-Net...")
    return model

class Data(Dataset):
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        dict_path = set_type+'.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())  
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, self.mm_data[k]['image'])
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        str_rr = self.mm_data[k]['report']
        input_ids = self.tokenizer.encode(str_rr, add_special_tokens=True, return_tensors='pt')
        outputs = self.bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state

        padding_length = tk_lim - last_hidden_state.shape[1]
        if padding_length>0:
            padding_token = self.tokenizer.pad_token_id
            padding_tensor = torch.full((1, padding_length, last_hidden_state.shape[2]), padding_token)
            padded_outputs = torch.cat([last_hidden_state, padding_tensor], dim=1)
        else:
            padded_outputs = last_hidden_state
        
        rr_vector = padded_outputs[:, :tk_lim, :] 

        rr = torch.tensor(rr_vector, dtype=torch.float32)                   # the report feature
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()  # the demographics information (age and sex)
        img_fea = torch.from_numpy(self.mm_data[k]['bts']).float()          # the image feature, such as shape and echo
        return img, label, rr, demo, img_fea

def train(args, model_para_path=None):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["LLNM_Net"]
    llnm_net = LLNM_Net(config, 224, zero_head=True, num_classes=num_classes)
    if model_para_path:
        llnm_net = load_weights(llnm_net, model_para_path)
    for param in llnm_net.parameters():
        param.requires_grad = True                                          # set requires_grad to True
    img_dir = args.DATA_DIR

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    train_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['train'])

    trainloader = DataLoader(train_data, batch_size=args.BSZ, shuffle=False, num_workers=0, pin_memory=True)

    optimizer_llnm_net = torch.optim.AdamW(llnm_net.parameters(), lr=3e-5, weight_decay=0.01)
    llnm_net, optimizer_llnm_net = amp.initialize(llnm_net.cuda(), optimizer_llnm_net, opt_level="O1")

    llnm_net = torch.nn.DataParallel(llnm_net)

    #----- Train ------
    print('--------Start training-------')
    num_epochs = 300
    loss_min = args.loss_min
    loss_fct = BCEWithLogitsLoss()
    loss_list, auc_list = [], []

    llnm_net.train()
    for epoch in range(num_epochs):
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        running_loss = 0.0
        for data in tqdm(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, rr, demo, img_fea = data
            rr = rr.view(-1, tk_lim, rr.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            img_fea = img_fea.view(-1, img_fea.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer_llnm_net.zero_grad()  

            with torch.set_grad_enabled(True):
                outputs = llnm_net(imgs, rr, img_fea, sex, age)         # logits, attn_weights, torch.mean(x, dim=1)
                preds = outputs[0]
                probs = torch.sigmoid(preds)

                target = labels.float()
                target_one_hot = torch.zeros(len(target), 2).cuda(non_blocking=True)
                target_one_hot.scatter_(1, target.long().unsqueeze(1), 1)

                loss = loss_fct(preds.view(-1, num_classes), target_one_hot)
                loss.backward()  # loss
                optimizer_llnm_net.step()

            running_loss += loss.item() * len(target)

            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)
  
        outGT = outGT.cpu().detach().numpy()
        outPRED = outPRED.cpu()
        # print("----data----")
        # print(outGT)
        outPred = torch.softmax(outPRED,dim=1).detach().numpy()
        outPred = np.argmax(outPred, axis=1)
        auc = roc_auc_score(outGT, outPred)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Auc: {auc:.4f}')

        if loss.item()<loss_min:
            loss_min = loss.item()
            model_para_path = 'model_epoch'+str(epoch)+'_bs'+str(args.BSZ)+'_loss'+str(round(loss_min, 3))+'_auc'+str(round(auc, 3))+'.pth'
            torch.save(llnm_net.state_dict(), model_para_path)

        loss_list.append(running_loss)
        auc_list.append(epoch)
        plt.figure()
        plt.title("Accuracy & AUC vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss or AUC")
        plt.plot(range(1, epoch + 2), loss_list, label="Loss")
        plt.plot(range(1, epoch + 2), auc_list, label="AUC")
        # plt.ylim((0, 1.))
        # plt.xticks(np.arange(1, num_epochs + 1, 10.0))
        plt.legend()
        plt.savefig("loss_auc.png")

        plt.figure()
        plt.title("Pred. & GrouTruth.")
        plt.xlabel("data index")
        plt.ylabel("label")
        plt.scatter(range(1, len(outGT) + 1), outGT, c='g', label="GroundTruth")
        plt.scatter(range(1, len(outPred) + 1), outPred, c='r', label="Prediction")
        # plt.ylim((0, 1.))
        # plt.xticks(np.arange(1, num_epochs + 1, 10.0))
        plt.legend()
        plt.savefig("pre_gt.png")

    torch.cuda.empty_cache()

    return model_para_path, loss_min
        
def test(args, model_para_path):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, 224, zero_head=True, num_classes=num_classes)
    llnm_net = load_weights(model, model_para_path)
    img_dir = args.DATA_DIR

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    test_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])

    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=0, pin_memory=True)

    optimizer_llnm_net = torch.optim.AdamW(llnm_net.parameters(), lr=3e-5, weight_decay=0.01)
    llnm_net, optimizer_llnm_net = amp.initialize(llnm_net.cuda(), optimizer_llnm_net, opt_level="O1")

    llnm_net = torch.nn.DataParallel(llnm_net)

    #----- Test ------
    print('--------Start testing-------')
    llnm_net.eval()
    with torch.no_grad():
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        for data in tqdm(testloader):
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, rr, demo, img_fea = data
            rr = rr.view(-1, tk_lim, rr.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            img_fea = img_fea.view(-1, img_fea.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            preds = llnm_net(imgs, rr, img_fea, sex, age)[0]
            probs = torch.sigmoid(preds)

            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)
  
        outGT = outGT.cpu().detach().numpy()
        outPRED = outPRED.cpu()
        outPred = torch.softmax(outPRED,dim=1).detach().numpy()
        outPred = np.argmax(outPred, axis=1)
        aurocMean = roc_auc_score(outGT, outPred)
        
        print('mean AUROC:' + str(aurocMean))
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)             # number of classes
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)             # batch size.
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)   # location of the imaging data.
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)   # file name of the clinical textual data (***.pkl).
    parser.add_argument('--loss_min', action='store', dest='loss_min', type=float)
    args = parser.parse_args()
    args.loss_min = 100.0
    model_para_path, args.loss_min = train(args)
    test(args,model_para_path)
