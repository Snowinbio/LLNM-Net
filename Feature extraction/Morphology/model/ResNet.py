import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score, accuracy_score
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from pretrainedmodels import inceptionv4, inceptionresnetv2


def get_resnet_model(num_classes=2, model_type='inceptionv4'):
    if model_type == 'inceptionv4':
        model = inceptionv4(pretrained='imagenet')
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)
    elif model_type == 'inceptionresnetv2':
        model = inceptionresnetv2(pretrained='imagenet')
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path, num_classes=2, model_type='inceptionv4'):
    model = get_resnet_model(num_classes=num_classes, model_type=model_type)
    model.load_state_dict(torch.load(model_path))
    return model


def train_model(model, model_name, dataloaders, criterion, optimizer, num_epochs=50, device='cpu'):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    test_acc_history = []
    train_auc_history = []
    val_auc_history = []
    test_auc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_prob = []
            running_label = []

            for inputs, labels, paths in dataloaders[phase]:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_prob.extend(outputs[:, 1].cpu().detach().numpy())
                running_label.extend(labels.cpu().detach().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_auc = roc_auc_score(running_label, running_prob)

            print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auc))

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_auc_history.append(epoch_auc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_name)
                best_auc = epoch_auc
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_auc_history.append(epoch_auc)

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} AUC: {:4f}'.format(best_acc, best_auc))
    model.load_state_dict(best_model_wts)
    visualize_results(train_acc_history, val_acc_history, test_acc_history, train_auc_history, val_auc_history,
                      test_auc_history, num_epochs)

    return model


def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_corrects = 0
    running_prob = []
    running_label = []
    # running_probs = []
    running_paths = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader['test']:
            sorted_indices = sorted(range(len(paths)), key=lambda x: (
                0 if 'benign' in paths[x] else 1,
                int(os.path.basename(paths[x]).split('.')[0])
            ))
            sorted_paths = [paths[i] for i in sorted_indices]
            sorted_inputs = inputs[sorted_indices].to(device)
            sorted_labels = labels[sorted_indices].to(device)
            for path in sorted_paths:
                # print(path)
                running_paths.append(path)

            outputs = model(sorted_inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == sorted_labels.data)
            running_prob.extend(outputs[:, 1].cpu().detach().numpy())
            running_label.extend(sorted_labels.cpu().detach().numpy())

    accuracy = running_corrects.double() / len(dataloader['test'].dataset)
    auc = roc_auc_score(running_label, running_prob)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test AUC: {auc:.4f}')

    return running_prob, running_label


def visualize_results(train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, num_epochs):
    tracc = [h.cpu().numpy() for h in train_acc]
    vacc = [h.cpu().numpy() for h in val_acc]
    tsacc = [h.cpu().numpy() for h in test_acc]

    trauc = [h for h in train_auc]
    vauc = [h for h in val_auc]
    tsauc = [h for h in test_auc]

    plt.title("Accuracy & AUC vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy or AUC")
    plt.plot(range(1, num_epochs + 1), tracc, label="Train Acc")
    plt.plot(range(1, num_epochs + 1), vacc, label="Val Acc")
    plt.plot(range(1, num_epochs + 1), tsacc, label="Test Acc")
    plt.plot(range(1, num_epochs + 1), trauc, label="Train AUC")
    plt.plot(range(1, num_epochs + 1), vauc, label="Val AUC")
    plt.plot(range(1, num_epochs + 1), tsauc, label="Test AUC")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 10.0))
    plt.legend()
    # plt.show()
