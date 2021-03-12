# Convolutional Auto-encoder

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
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.dconv2 = nn.ConvTranspose2d(16, 1, 3, padding=1, stride=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, encode):
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.dconv1(encode))
        x = self.upsample1(x)

        x = F.relu(self.dconv2(x))
        x = self.upsample2(x)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(x)

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
