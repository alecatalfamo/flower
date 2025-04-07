import torch 
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import Compose
from typing import Union
from typing import List
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter

def train_local(net: nn.Module, 
                trainloader: DataLoader, 
                epochs: int, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                DEVICE: torch.device):
    
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx,(images, labels) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if batch_idx % 100 == 0:
                print(f"Image number {total} of {len(trainloader.dataset)}", end="\r", flush=True)
            
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            #test_acc, test_loss = evaluate(net, testloader, criterion, DEVICE)
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}", end="\r", flush=True)
