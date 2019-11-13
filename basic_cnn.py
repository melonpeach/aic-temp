from torch import nn
import torch.nn.functional as F

class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5) # in_ch, out_ch, filter_size, stride
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # input: 28*28
        x = F.relu(self.conv1(x))   # 24*24*8
        x = self.pool(x)            # 12*12*8
        x = F.relu(self.conv2(x))   # 8*8*16
        x = self.pool(x)            # 4*4*16
        x = self.conv3(x)           # 2*2*32
        conv_out = F.relu(x)
        conv_out = self.flatten(conv_out)
        fc_out = F.relu(self.fc1(conv_out))
        fc_out = F.relu(self.fc2(fc_out))
        fc_out = self.fc3(fc_out)

        #return F.softmax(fc_out)
        return fc_out
    
    def flatten(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s
        
        return x.view(-1, num_features)