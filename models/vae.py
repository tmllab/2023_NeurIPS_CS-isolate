import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from .encoders import *
from .PreResNet import *


__all__ = ["VAE_FASHIONMNIST","VAE_CIFAR10","VAE_CIFAR100"]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            index: Tensor, shape ``[batch_size]``
        """
        return self.pe[index]


class BaseVAE(nn.Module):
    def __init__(self, feature_dim=28, num_hidden_layers=1, hidden_size=32, z_dim =10, num_classes=100, embed_size = 10):
        super().__init__()
        self.zs_encoder = Z_Encoder(feature_dim=feature_dim, num_classes=num_classes, embed_size=embed_size, num_hidden_layers=num_hidden_layers, hidden_size = hidden_size, z_dim=z_dim)
        self.x_decoder = X_Decoder(feature_dim=feature_dim, num_hidden_layers=num_hidden_layers, num_classes=num_classes, embed_size=embed_size, hidden_size = hidden_size, z_dim=2*z_dim)
        self.kl_divergence = None
        self.flow  = None
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.replay_num = 1000+10
        self.zc_prior = nn.Sequential(nn.Linear(32, 128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128,128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128, z_dim))
        self.zs_prior = nn.Sequential(nn.Linear(self.embed_size,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, z_dim))
        self.pe_uc = PositionalEncoding(32, 80000)
        self.pe_us = PositionalEncoding(self.embed_size, self.replay_num)

    def _y_hat_reparameterize(self, c_logits):
        return F.gumbel_softmax(c_logits, dim=1)

    def _z_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor, uc: torch.Tensor, us: torch.Tensor, net): 
        zc_mean, zc_logvar, zc, c_logits = net.forward(x)
        embed_uc = self.pe_uc(uc)
        embed_us = self.pe_us(us)
        p_zc_m = self.zc_prior(embed_uc)
        p_zs_m = self.zs_prior(embed_us)
        zs_mean, zs_logvar  = self.zs_encoder(x)
    
        zs = self._z_reparameterize(zs_mean, zs_logvar)
        z = torch.cat((zc, zs),dim=1)
        x_hat = self.x_decoder(z)

        return x_hat, c_logits, zc_mean, zc_logvar, zs_mean, zs_logvar, p_zc_m, p_zs_m



class VAE_FASHIONMNIST(BaseVAE):
    def __init__(self, feature_dim=28, input_channel=1, z_dim =10, num_classes=10, embed_size = 10):
        super().__init__()
        
        self.zs_encoder = CONV_Encoder_FMNIST(feature_dim=feature_dim, num_classes=num_classes, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_FMNIST(num_classes=num_classes, z_dim=2*z_dim)
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.replay_num = 1000+10
        self.zc_prior = nn.Sequential(nn.Linear(32, 128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128,128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128, z_dim))
        self.zs_prior = nn.Sequential(nn.Linear(self.embed_size,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, z_dim))
        self.pe_uc = PositionalEncoding(32, 80000)
        self.pe_us = PositionalEncoding(self.embed_size, self.replay_num)


class VAE_CIFAR100(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim =32, num_classes=100, embed_size = 10):
        super().__init__()
        self.zs_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, embed_size=embed_size, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=2*z_dim)
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.replay_num = 1000+10
        self.zc_prior = nn.Sequential(nn.Linear(32, 128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128,128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128, z_dim))
        self.zs_prior = nn.Sequential(nn.Linear(self.embed_size,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, z_dim))
        self.pe_uc = PositionalEncoding(32, 80000)
        self.pe_us = PositionalEncoding(self.embed_size, self.replay_num)


class VAE_CIFAR10(BaseVAE):
    def __init__(self, feature_dim=32, input_channel=3, z_dim =32, num_classes=10, embed_size = 10):
        super().__init__()
        self.zs_encoder = CONV_Encoder_CIFAR(feature_dim=feature_dim, num_classes=num_classes, embed_size=embed_size, z_dim=z_dim)
        self.x_decoder = CONV_Decoder_CIFAR(num_classes=num_classes, z_dim=2*z_dim)
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.replay_num = 1000+10
        self.zc_prior = nn.Sequential(nn.Linear(32, 128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128,128),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(128, z_dim))
        self.zs_prior = nn.Sequential(nn.Linear(self.embed_size,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64,64),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(64, z_dim))
        self.pe_uc = PositionalEncoding(32, 80000)
        self.pe_us = PositionalEncoding(self.embed_size, self.replay_num)
