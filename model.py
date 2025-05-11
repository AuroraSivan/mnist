# model.py
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输出 [32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 [32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输出 [64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 [64, 7, 7]
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
