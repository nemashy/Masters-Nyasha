import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1, 256)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class Model2(nn.Module):
    """Batch norm model"""

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        # Dense layer
        self.fc1 = nn.Linear(32 * 14 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1((self.conv1(x)))))
        x = self.pool(F.relu(self.bn2((self.conv2(x)))))
        x = self.pool(F.relu(self.bn3((self.conv3(x)))))
        x = x.view(-1, 32 * 14 * 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Model3(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        # Global average pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layer 1
        self.fc1 = nn.Linear(32 * 1 * 1, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.avgpool(x)

        x = x.view(-1, 32 * 1 * 1)

        x = self.fc1(x)

        return x


# ============= Residual Network =====================

# Residual network model
class FirstLayer(nn.Module):
    """The layer of the network"""

    def __init__(self, num_channels=32):
        super(FirstLayer, self).__init__()

        # First convolutional layer of the model
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool(out)

        return out


class Block(nn.Module):
    """A block that will be skipped over by the residual connectionS"""

    def __init__(self, num_channels=32):
        super(Block, self).__init__()
        self.num_channels = num_channels

        # First layer of the block
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Second layer of the block
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out += identity  # Creating the skip connection

        return out


class Model4(nn.Module):  # Enerst Net
    def __init__(self, num_labels=6):
        super(Model4, self).__init__()
        # Network = First layer -> block1 -> block2 ....
        self.first_layer = FirstLayer()
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.output_channels = self.block3.num_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_labels)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)
        return out
