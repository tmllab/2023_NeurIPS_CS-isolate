import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
__all__ = ["CONV_Decoder_FMNIST", "CONV_Encoder_FMNIST", "Z_Encoder", "X_Decoder","CONV_Encoder_CIFAR","CONV_Decoder_CIFAR"]


def make_hidden_layers(num_hidden_layers=1, hidden_size=5, prefix="y"):
    block = nn.Sequential()
    for i in range(num_hidden_layers):
        block.add_module(prefix+"_"+str(i), nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.BatchNorm1d(hidden_size),nn.LeakyReLU()))
    return block


class CONV_Encoder_FMNIST(nn.Module):
    def __init__(self, in_channels =1, feature_dim = 28, num_classes = 2,  hidden_dims = [32, 64, 128, 256], z_dim = 2):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*4, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


class CONV_Decoder_FMNIST(nn.Module):

    def __init__(self, num_classes = 2, hidden_dims = [256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               ),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 4))


    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out



class CONV_Encoder_CIFAR(nn.Module):
    def __init__(self, in_channels =3, feature_dim = 32, num_classes = 2, embed_size = 10,  hidden_dims = [32, 64, 128, 256], z_dim = 2):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.in_channels = in_channels
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(nn.Linear(hidden_dims[-1]*4,hidden_dims[-1]*4),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(hidden_dims[-1]*4, z_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(hidden_dims[-1]*4,hidden_dims[-1]*4),
                                                                    nn.LeakyReLU(),
                                                                    nn.Linear(hidden_dims[-1]*4, z_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var



class CONV_Decoder_CIFAR(nn.Module):

    def __init__(self, num_classes = 2, embed_size=10, hidden_dims = [256, 128, 64,32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)


        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1))


    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out



class Z_Encoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, embed_size=10, num_hidden_layers=1, hidden_size = 5, z_dim = 2):
        super().__init__()
        self.z_fc1 = nn.Linear(feature_dim, hidden_size)
        self.z_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="z")
        self.z_fc_mu = nn.Linear(hidden_size, z_dim)  # fc21 for mean of Z
        self.z_fc_logvar = nn.Linear(hidden_size, z_dim)  # fc22 for log variance of Z

    def forward(self, x):
        out = F.leaky_relu(self.z_fc1(x))
        out = self.z_h_layers(out)
        mu = F.elu(self.z_fc_mu(out))
        logvar = F.elu(self.z_fc_logvar(out))
        return mu, logvar


class X_Decoder(nn.Module):
    def __init__(self, feature_dim = 2, num_classes = 2, embed_size=10, num_hidden_layers=1, hidden_size = 5, z_dim = 1):
        super().__init__()
        self.recon_fc1 = nn.Linear(z_dim, hidden_size)
        self.recon_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="recon")
        self.recon_fc2 = nn.Linear(hidden_size, feature_dim)    

    def forward(self, z):
        out = F.leaky_relu(self.recon_fc1(z))
        out = self.recon_h_layers(out)
        x = self.recon_fc2(out)
        return x