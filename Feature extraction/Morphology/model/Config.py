import torch.optim as optim
import torch.nn as nn

class Config:
    @staticmethod
    def edge_model_parameters(model):
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
        criterion = nn.CrossEntropyLoss()
        num_epochs = 100
        return optimizer, criterion, num_epochs

    @staticmethod
    def texture_model_parameters(model):
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
        criterion = nn.CrossEntropyLoss()
        num_epochs = 100
        return optimizer, criterion, num_epochs
