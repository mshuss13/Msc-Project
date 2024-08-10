import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # (32, 148, 148)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 74, 74)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # (64, 72, 72)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64, 36, 36)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),  # (128, 34, 34)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (128, 17, 17)

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),  # (128, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # (128, 5, 5)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output from the conv layers
            nn.Linear(in_features=128 * 5 * 5, out_features=512),  # Adjust the input size accordingly
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),  # Final layer output to N_TYPES classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flattening features (batch_size, features)
        x = self.fc_layers(x)
        return x
