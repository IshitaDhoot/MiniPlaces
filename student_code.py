# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class SimpleFCNet(nn.Module):
    """
    A simple neural network with fully connected layers
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        super(SimpleFCNet, self).__init__()
        # create the model by adding the layers
        layers = []

        ###################################
        #     fill in the code here       #
        ###################################
        # Add a Flatten layer to convert the 2D pixel array to a 1D vector
        layers.append(nn.Flatten())

        # Add a fully connected / linear layer with 128 nodes
        layers.append(nn.Linear(input_shape[0]*input_shape[1], 128))

        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))

        # Append a fully connected / linear layer with 64 nodes
        layers.append(nn.Linear(128, 64))

        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))

        # Append a fully connected / linear layer with num_classes (10) nodes
        layers.append(nn.Linear(64, num_classes))

        self.layers = nn.Sequential(*layers)

        self.reset_params()

    def forward(self, x):
        # the forward propagation
        out = self.layers(x)
        if self.training:
            # softmax is merged into the loss during training
            return out
        else:
            # attach softmax during inference
            out = nn.functional.softmax(out, dim=1)
            return out

    def reset_params(self):
        # to make our model a faithful replica of the Keras version
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class SimpleConvNet(nn.Module):
    """
    A simple convolutional neural network
    """
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(SimpleConvNet, self).__init__()
        ####################################################
        # you can start from here and create a better model
        ####################################################
        
        # Maxpool 2x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layers with batch norm
        
        self.conv1 = nn.Conv2d(3, 8, 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 8, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(8)
               
        self.conv3 = nn.Conv2d(8, 32, 3, padding = 1)
        self.norm3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, 3, padding = 1)
        self.norm4 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 64, 3, padding = 1)
        self.norm5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 64, 3, padding = 1)
        self.norm6 = nn.BatchNorm2d(64)
        
        self.conv7 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm7 = nn.BatchNorm2d(128)
        
        self.conv8 = nn.Conv2d(128, 128, 3, padding = 1)
        self.norm8 = nn.BatchNorm2d(128)
        
        # fully connected layer with batch norm

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.norm9 = nn.BatchNorm1d(128)
       
        self.fc2 = nn.Linear(128, 64)
        self.norm10 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 100)

    def forward(self, x):
        
        out = F.elu(self.norm1(self.conv1(x)))
        out = F.elu(self.norm2(self.conv2(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm3(self.conv3(out)))
        out = F.elu(self.norm4(self.conv4(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm5(self.conv5(out)))
        out = F.elu(self.norm6(self.conv6(out)))
        out = self.pool(out)
        
        out = F.elu(self.norm7(self.conv7(out)))
        out = F.elu(self.norm8(self.conv8(out)))
        
        out = out.view(-1, 128 * 4 * 4)
        
        out = F.elu(self.norm9(self.fc1(out)))
        out = F.elu(self.norm10(self.fc2(out)))
        out = self.fc3(out)

        return out

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ######################################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ######################################################

        # 1) zero the parameter gradients

        # 2) forward + backward + optimize


        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        # train_loss += loss.item()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
