import torch.nn as nn
import torch.nn.functional as F

class Us_CNN(nn.Module):
    def __init__(self, channels, num_classes=2):
        super(Us_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.deconv1 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=15, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=15, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=13, kernel_size=13, stride=2, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.expand(-1, -1, 10)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        return x
    
class CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x

