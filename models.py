import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self, log_dir):
        super(BaseModel, self).__init__()
        dir_name = os.path.join(log_dir, 'logs/')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open(str(dir_name) + st, 'w')
    
    def log(self, str):
        #print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        #return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr # TODO: Implement decreasing learning rate's rules
        lr = lr * pow(0.9, epoch/50) 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
       


class LazyNet(BaseModel):
    def __init__(self, log_dir, device):
        super(LazyNet, self).__init__(log_dir)
        # TODO: Define model here
        self.lin1 = nn.Linear(32*32*3, 10)

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        # Flatten from image to an array
        x = x.view(-1, 32*32*3);
        # Do the forward pass
        x = self.lin1(x)
        return x;
        

class BoringNet(BaseModel):
    def __init__(self, log_dir, device):
        super(BoringNet, self).__init__(log_dir)
        # TODO: Define model here
        self.lin1 = nn.Linear(32*32*3, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)


    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        # Flatten from image to an array
        x = x.view(-1, 32*32*3);

        # Do the forward pass with activations
        # x = self.lin1(x)
        # x = F.relu(x)
        # x = self.lin2(x)
        # x = F.relu(x)
        # x = self.lin3(x)

        # Do the forward pass without activations
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x;


class CoolNet(BaseModel):
    def __init__(self, log_dir, device):
        super(CoolNet, self).__init__(log_dir)
        # TODO: Define model here
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
