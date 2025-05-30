import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from .Decoder import Decoder
from .FusionRNN import FusionRNN
from .ImageEncoder import ImageEncoder
from .IMUEncoder import IMUEncoder


# This will be the overarching class that connects all 4 of our NNs
class SLAMErrorPredictor(nn.Module):
    def __init__(
        self,
        seq_len=1,
        image_width=24,
        image_height=24,
        image_channels=1,
        image_embed_size=64,
        imu_input_size=6,
        imu_embed_size=64,
        imu_dropout=0.3,
        imu_hidden_size=128,
        fusion_output_dim=64,
        fusion_hidden_size=128,
        fusion_dropout=0.3,
        decoder_hidden_size=128,
    ):
        super(SLAMErrorPredictor, self).__init__()
        self.seq_len = seq_len  # The number of image frames in the sequence - (N)

        self.W = image_width  # Default image width is 24
        self.H = image_height  # Default image height is 24
        self.C = image_channels  # Assumes grayscale by default
        self.I_e = image_embed_size  # Image embedding size

        self.M = imu_input_size  # IMU input size, default is 6 (3-axis accel + gyro)
        self.M_e = (  # IMU embedding size, if none is provided, defaults to M * 10 (10 IMU updates per image frame)
            imu_embed_size or self.M * 10
        )
        self.imu_dropout = imu_dropout
        self.imu_hidden_size = imu_hidden_size

        self.E = (  # Total embedding size for the fusion branch (2 images + IMU)
            self.I_e * 2 + self.M_e
        )
        self.fusion_dropout = (  # Default dropout for the fusion branch is 0.3
            fusion_dropout
        )
        self.fusion_hidden_size = fusion_hidden_size
        self.F = fusion_output_dim

        self.decoder_hidden_size = decoder_hidden_size  # Decoder hidden size

        self.imu_encoder = IMUEncoder(
            input_dim=self.M,
            hidden_dim=self.imu_hidden_size,
            output_dim=self.M_e,
            dropout=self.imu_dropout,
            window_size=self.seq_len * 10,
        )

        self.image_encoder_L = ImageEncoder(
            seq_len=self.seq_len,
            ch_in=self.C,
            out_dim=self.I_e,
        )

        self.image_encoder_R = ImageEncoder(
            seq_len=self.seq_len,
            ch_in=self.C,
            out_dim=self.I_e,
        )

        self.fusion = FusionRNN(
            input_dim=self.E,  # E is the summed embedding size of the 3 encoders (I_e * 2 + M_e)
            output_dim=self.F,
            hidden_dim=self.fusion_hidden_size,
            bidirectional=False,
            dropout=self.fusion_dropout,
        )

        self.decoder = Decoder(
            input_dim=self.F,
            hidden_dim=self.decoder_hidden_size,  # Decoder hidden size
            output_dim=7,  # Output is a pose delta (Δx, Δy, Δz, Δq)
            dropout=self.fusion_dropout,
        )

    def forward(self, x, prediction_len=10):
        """
        Assuming data pipeline feeds us a list of 3 inputs: L img, R img, IMU data, in that order of a list. Each image comes in after 10 IMU update cycles.
        image shape: [B,N,C,H,W], N = # of 20 Hz updates (seq_len)
        imu data: [B,10N,6]

        Once fed into encoders, expected cat shape [B, N, E], where E is the summed embedding sizes for the 3 encoder outputs. Inputs into fusion branch
        which will output a shape [B, ..., ...] which is then fed into the decoder.

        The decoder will output K future pose readings (prediction_len) with shape [B, K, 7]
        """
        imu = x[0]  # [B, 10N, M]
        img_L = x[1]  # [B, N, C, H, W]
        img_R = x[2]  # [B, N, C, H, W]


        V_L = self.image_encoder_L(img_L)  # [B, N, I_e]
        V_R = self.image_encoder_R(img_R)  # [B, N, I_e]
        I = self.imu_encoder(imu)  # [B, N, M_e]

        assert (
            V_L.shape[0:-1] == V_R.shape[0:-1] == I.shape[0:-1]
        ), "Encoder shape mismatch"

        encoded_features = torch.cat(  # [B, N, E] where E is the summed embedding size of the 3 encoders (I_e * 2 + M_e)
            (V_L, V_R, I), -1
        )
        full_encoding = self.fusion(  # [B, N, F] where F is the fusion hidden size
            encoded_features
        )
        out = self.decoder(  # [B, 1, 7], Estimate of the current pose at the end of the sequence
            full_encoding, prediction_len
        )
        return out
