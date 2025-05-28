import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from Decoder import Decoder
from FusionRNN import FusionRNN
from ImageEncoder import ImageEncoder
from IMUEncoder import IMUEncoder

# This will be the overarching class that connects all 4 of our NNs
class SLAMErrorPredictor(nn.Module):
    def __init__(self, seq_len, image_embed_size=256, imu_embed_size=256, loss_fnc=None, lr=1e-3, weight_decay=0):
        super(SLAMErrorPredictor, self).__init__()        
        self.seq_len = seq_len  # N
        self.loss_fnc = loss_fnc or nn.MSELoss()

        # TODO object arguments
        self.decoder = Decoder()
        self.fusion = FusionRNN()
        self.image_encoder_L = ImageEncoder(seq_len, output_dim=image_embed_size)
        self.image_encoder_R = ImageEncoder(seq_len, output_dim=image_embed_size)
        self.imu_encoder = IMUEncoder()

    def forward(self, x, prediction_len):
        """
        Assuming data pipeline feeds us a list of 3 inputs: L img, R img, IMU data, in that order of a list. Each image comes in after 10 IMU update cycles.
        image shape: [B,N,C,H,W], N = # of 20 Hz updates (seq_len)
        imu data: [B,10N,7]?

        Once fed into encoders, expected cat shape [B, N, E], where E is the summed embedding sizes for the 3 encoder outputs. Inputs into fusion branch
        which will output a shape [B, ..., ...] which is then fed into the decoder.

        The decoder will output K future pose readings (prediction_len) with shape [B, K, 7]
        """
        img_L = x[0]
        img_R = x[1]
        imu = x[2]

        V_L = self.image_encoder_L(img_L)
        V_R = self.image_encoder_R(img_R)
        I = self.imu_encoder(imu)

        assert V_L.shape[0:-1] == V_R.shape[0:-1] == I.shape[0:-1], "Encoder shape mismatch"

        encoded_features = torch.cat((V_L, V_R, I), -1)  # [B, N, E]
        full_encoding = self.fusion(encoded_features)  # [B, N, F]?
        out = self.decoder(full_encoding, prediction_len)  # [B, K, 7], K future pose readings, of which there are 7 values
        return out