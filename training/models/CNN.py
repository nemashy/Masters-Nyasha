# Convlutional Neural Network
from torch.quantization import QuantStub, DeQuantStub
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        #self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 30 * 9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 30 * 9)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

class ConvNetX1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 63 * 21, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 63 * 21)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

class ConvNetX2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        #self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 14 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 14 * 3)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


class ConvNetX3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        #self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 29 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 29 * 8)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x



class ConvNetX4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv5 = nn.Conv2d(32, 32, 3)
        #self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 28 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 32 * 28 * 7)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


class ConvNetX5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 14 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 14 * 3)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ConvNetX6(nn.Module):
    """Batch norm model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
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


# Convolutional neural network (two convolutional layers)
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=2),
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
        self.fc1 = nn.Sequential(nn.Linear(______, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Convolutional neural network (two convolutional layers)
class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 64, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Batch Norm Model !!!!!!!!!! Correcto
class ConvNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 3
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layer 1
        self.fc1 = nn.Linear(32 * 1 * 1, 6)
        # self.bn3 = nn.BatchNorm1d(128)

        # Dense layer 2
        # self.fc2 = nn.Linear(128, 64)
        # self.bn4 = nn.BatchNorm1d(64)
        # self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Global average pool
        x = self.avgpool(x)

        x = x.view(-1, 32 * 1 * 1)

        # x = self.avgpool(x)

        x = self.fc1(x)

        # x = self.dropout(x)
        # x = F.relu(self.bn3(self.fc1(x)))
        # # x = self.dropout(x)
        # x = F.relu(self.bn4(self.fc2(x)))
        # # x = self.dropout(x)
        # x = self.fc3(x)
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


class ErnNet(nn.Module):  # Enerst Net
    def __init__(self, num_labels=6):
        super(ErnNet, self).__init__()
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


class ErnNet2(nn.Module):  # Enerst Net
    def __init__(self, num_labels=6):
        super(ErnNet2, self).__init__()
        # Network = First layer -> block1 -> block2 ....
        self.first_layer = FirstLayer()
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.block4 = Block()
        self.output_channels = self.block4.num_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_labels)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)

        return out



class ErnNetX1(nn.Module):  # Enerst Net
    def __init__(self, num_labels=6):
        super(ErnNetX1, self).__init__()
        # Network = First layer -> block1 -> block2 ....
        self.first_layer = FirstLayer()
        self.block1 = Block()
        # self.block2 = Block()
        # self.block3 = Block()
        self.output_channels = self.block1.num_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_labels)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)
        return out


class ErnNetX2(nn.Module):  # Enerst Net
    def __init__(self, num_labels=6):
        super(ErnNetX2, self).__init__()
        # Network = First layer -> block1 -> block2 ....
        self.first_layer = FirstLayer()
        self.block1 = Block()
        self.block2 = Block()
        # self.block3 = Block()
        self.output_channels = self.block2.num_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_labels)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        # out = self.block3(out)
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)
        return out

class ErnNetX4(nn.Module):  # Enerst Net
    def __init__(self, num_labels=6):
        super(ErnNetX4, self).__init__()
        # Network = First layer -> block1 -> block2 ....
        self.first_layer = FirstLayer()
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.block4 = Block()
        self.output_channels = self.block2.num_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_labels)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)
        return out

# ============= Residual Network End =============


# ============= Plain CNN =============

# Batch Norm Model !!!!!!!!!! Correcto
class Net(nn.Module):
    def __init__(self, input_channels=16, num_classes=6):
        super(Net, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, input_channels, 3)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2
        self.conv2 = nn.Conv2d(input_channels, 2 * input_channels, 3)
        self.bn2 = nn.BatchNorm2d(2 * input_channels)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3
        self.conv3 = nn.Conv2d(2 * input_channels, 4 * input_channels, 3)
        self.bn3 = nn.BatchNorm2d(4 * input_channels)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global average pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layer 1
        self.output_channels = 4 * input_channels
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_classes)
        # self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.avgpool(out)

        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)

        return out


# ============= Plain CNN End =============


class ErnNetQAT(nn.Module):
    def __init__(self, num_labels=6):
        super(ErnNetQAT, self).__init__()
        # Network = First layer -> block1 -> block2 ....
        self.quant = QuantStub()
        self.first_layer = FirstLayer()
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.output_channels = self.block3.num_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels * 1 * 1, num_labels)
        self.dequant = DeQuantStub()

    def forward(self, x):
        out = self.quant(x)
        out = self.first_layer(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = out.view(-1, self.output_channels * 1 * 1)
        out = self.fc1(out)

        out = self.dequant(out)
        return out
