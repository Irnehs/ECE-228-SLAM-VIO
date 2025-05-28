import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from Decoder import Decoder
from FusionRNN import FusionRNN
from ImageEncoder import ImageEncoder
from IMUEncoder import IMUEncoder

# This will be the overarching class that connects all 4 of our NNs
class SLAMPathError(nn.Module):
    def __init__(self):
        super(SLAMPathError, self).__init__()

        # TODO object arguments
        self.decoder = Decoder()
        self.fusion = FusionRNN()
        self.image_encoder_L = ImageEncoder(output_dim=128)
        self.image_encoder_R = ImageEncoder(output_dim=128)
        self.imu_encoder = IMUEncoder()

    def forward(self, x):
        """
        Assuming data pipeline feeds us a list of 3 inputs: L img, R img, IMU data
        image shape: [B,C,H,W]
        imu data: [B,6,10]?
        """
        img_L = x[0]
        img_R = x[1]
        imu = x[2]

        V_L = self.image_encoder_L(img_L)
        V_R = self.image_encoder_R(img_R)
        I = self.imu_encoder(imu)

        encoded_features = torch.cat((V_L, V_R, I), 0)
        full_encoding = self.fusion(encoded_features)
        out = self.decoder(full_encoding)
        return out