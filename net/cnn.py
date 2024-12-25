import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


# class Us_CNN(nn.Module):
#     def __init__(self, channels, num_classes=2):
#         super(Us_CNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.deconv1 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=15, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=15, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=13, kernel_size=13, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm1d(32)  # 为 conv1 之后添加批归一化
#         self.bn2 = nn.BatchNorm1d(64)  # 为 conv2 之后添加批归一化
#         self.bn3 = nn.BatchNorm1d(32)  # 为 conv3 之后添加批归一化
#         self.bn_deconv1 = nn.BatchNorm1d(64)  # 为 deconv1 之后添加批归一化
#         self.bn_deconv2 = nn.BatchNorm1d(32)  # 为 deconv2 之后添加批归一化
#         self.bn_deconv3 = nn.BatchNorm1d(13)  # 为 deconv3 之后添加批归一化


#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))  # 在 conv1 之后添加批归一化
#         x = F.relu(self.bn2(self.conv2(x)))  # 在 conv2 之后添加批归一化
#         x = F.relu(self.bn3(self.conv3(x)))  # 在 conv3 之后添加批归一化
#         x = self.global_avg_pool(x)
#         x = x.expand(-1, -1, 10)
#         x = F.relu(self.bn_deconv1(self.deconv1(x)))  # 在 deconv1 之后添加批归一化
#         x = F.relu(self.bn_deconv2(self.deconv2(x)))  # 在 deconv2 之后添加批归一化
#         x = F.relu(self.bn_deconv3(self.deconv3(x)))  # 在 deconv3 之后添加批归一化

#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class Us_CNN(nn.Module):
    def __init__(self, channels, num_classes=2):
        super(Us_CNN, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))  # 添加可训练的参数 alpha
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose1d(in_channels=32, out_channels=13, kernel_size=13, stride=2, padding=2),
            nn.BatchNorm1d(13),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        # 使用 alpha 对输入进行缩放，这里可以根据实际需求修改使用方式
        x = x * self.alpha
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.expand(-1, -1, 10)
        x = self.deconv(x)
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

