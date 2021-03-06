# Convolutional Auto-encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model 1
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ## encoder layers ##

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # add second hidden layer
        """"
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # compressed representation
        """

        return x


class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(8, 16, 3, padding=1, stride=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dconv2 = nn.ConvTranspose2d(16, 1, 3, padding=1, stride=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, encode):
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.dconv1(encode))
        x = self.upsample1(x)

        x = F.relu(self.dconv2(x))
        x = self.upsample2(x)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(x)

        return x


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)

        return x


class Encoder_2(nn.Module):
    def __init__(self):
        super(Encoder_2, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        return x


class Decoder_2(nn.Module):
    def __init__(self):
        super(Decoder_2, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, encode):

        x = self.deconv3(encode)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.upsample2(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class CAE_2(nn.Module):
    def __init__(self):
        super(CAE_2, self).__init__()

        self.encoder = Encoder_2()
        self.decoder = Decoder_2()

    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)

        return x


## Model 3


class Encoder_3(nn.Module):
    def __init__(self):
        super(Encoder_3, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        return x


class Decoder_3(nn.Module):
    def __init__(self):
        super(Decoder_3, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, encode):

        x = self.deconv3(encode)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.upsample2(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class CAE_3(nn.Module):
    def __init__(self):
        super(CAE_3, self).__init__()

        self.encoder = Encoder_3()
        self.decoder = Decoder_3()

    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)

        return x


# Model 4


class Encoder_4(nn.Module):
    def __init__(self):
        super(Encoder_4, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.dense1 = nn.Linear(8192, 4096)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        x = x.view(-1, 8192)
        x = self.dense1(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)

        return x


class Decoder_4(nn.Module):
    def __init__(self):
        super(Decoder_4, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dense1 = nn.Linear(4096, 8192)

        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, encode):
        x = self.dense1(encode)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)

        x = x.view(x.size(0), 8, 32, 32)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.upsample2(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = F.sigmoid(x)

        return x


class CAE_4(nn.Module):
    def __init__(self):
        super(CAE_4, self).__init__()

        self.encoder = Encoder_4()
        self.decoder = Decoder_4()

    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)

        return x


# CAE 5


class Encoder_5(nn.Module):
    def __init__(self):
        super(Encoder_5, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, 5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(50, 30, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)

    def forward(self, img):
        x = self.conv1(img)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        return x


class Decoder_5(nn.Module):
    def __init__(self):
        super(Decoder_5, self).__init__()

        self.deconv2 = nn.ConvTranspose2d(30, 50, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(50, 1, 5, stride=1, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, encode):

        x = self.deconv2(encode)
        x = F.relu(x)

        x = self.upsample1(x)

        x = self.deconv1(x)
        x = torch.sigmoid(x)

        return x


class CAE_5(nn.Module):
    def __init__(self):
        super(CAE_5, self).__init__()

        self.encoder = Encoder_5()
        self.decoder = Decoder_5()

    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)

        return x
