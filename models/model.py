import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import *
from pytorch3d.loss import chamfer_distance
from models.hyper_encoder import HyperEncoder
from models.hyper_decoder import HyperDecoder
from models.bitEstimator import BitEstimator
import math


class get_model(nn.Module):
    def __init__(self, normal_channel=False, bottleneck_size=256, use_hyperprior=True, recon_points=2048):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.bottleneck_size = bottleneck_size
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=False)

        self.feature_enhence1 = FeatureEnhencementModule(channels=256)

        self.feature_enhence2 = FeatureEnhencementModule(channels=256)

        self.use_hyper = use_hyperprior
        self.recon_points = recon_points

        self.decompression = ReconstructionLayer(recon_points // 16, bottleneck_size, 256)
        self.coor_reconstruction_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self.coor_upsampling1 = CoordinateUpsamplingModule(ratio=2, channels=256, radius=0.05)
        self.coor_upsampling2 = CoordinateUpsamplingModule(ratio=2, channels=256, radius=0.05)
        self.coor_upsampling3 = CoordinateUpsamplingModule(ratio=2, channels=256, radius=0.05)
        self.coor_upsampling4 = CoordinateUpsamplingModule(ratio=2, channels=256, radius=0.05)

        if use_hyperprior:
            self.he = HyperEncoder(bottleneck_size)
            self.hd = HyperDecoder(bottleneck_size // 32)
            self.bitEstimator_z = BitEstimator(bottleneck_size // 32)
        else:
            self.bitEstimator = BitEstimator(bottleneck_size)

    def forward(self, xyz, global_step=None):
        B, N, C = xyz.shape

        if self.normal_channel:
            l0_feature = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_feature = xyz
            l0_xyz = xyz

        pc_gd = l0_xyz

        l0_xyz = l0_xyz.permute(0, 2, 1)
        l0_feature = l0_feature.permute(0, 2, 1)
        l1_xyz, l1_feature = self.sa1(l0_xyz, l0_feature)
        l2_xyz, l2_feature = self.sa2(l1_xyz, l1_feature)
        x = self.feature_enhence1(l2_xyz.permute(0, 2, 1), l2_feature)

        x = F.adaptive_max_pool1d(x, 1).view(B, -1)

        if self.use_hyper:
            z = self.he(x)
            z_noise = torch.nn.init.uniform_(torch.zeros_like(z), -0.5, 0.5)
            if self.training:
                compressed_z = z + z_noise
            else:
                compressed_z = torch.round(z)
        x_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)

        if self.training:
            compressed_x = x + x_noise
        else:
            compressed_x = torch.round(x)

        if self.use_hyper:
            recon_sigma = self.hd(compressed_z)

        decoder_local_feature = self.decompression(compressed_x.unsqueeze(1))
        new_xyz0 = self.coor_reconstruction_layer(decoder_local_feature)

        new_feature0 = self.feature_enhence2(new_xyz0, decoder_local_feature.permute(0, 2, 1)).permute(0, 2, 1)

        new_xyz1, new_feature1 = self.coor_upsampling1(new_xyz0, new_feature0)
        new_xyz2, new_feature2 = self.coor_upsampling2(new_xyz1, new_feature1)
        new_xyz3, new_feature3 = self.coor_upsampling3(new_xyz2, new_feature2)
        new_xyz4, new_feature4 = self.coor_upsampling4(new_xyz3, new_feature3)

        coor_recon = new_xyz4

        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_x, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        bpp = (total_bits_z + total_bits_feature) / (B * N)
        cd = chamfer_distance(pc_gd, coor_recon)[0]

        return bpp, coor_recon, cd


class get_loss(nn.Module):
    def __init__(self, lam=1):
        super(get_loss, self).__init__()
        self.lam = lam

    def forward(self, bpp, cd_loss):
        return self.lam * cd_loss + bpp, cd_loss, bpp
