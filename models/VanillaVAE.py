import torch
import torch.nn as nn


class VariantAutoEncoder(nn.Module):
    def __init__(self):
        super(VariantAutoEncoder, self).__init__()
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512]
        latent_dim = 128
        self.latent_dim = latent_dim

        Encoder_Module = []
        for h_dim in hidden_dims:
            Encoder_Module.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*Encoder_Module)
        self.encoder_flatten = nn.Flatten()
        '''
            Cin  Cout  Sizein  Sizeout
            3    16    128     64
            16   32    64      32
            32   64    32      16
            64   128   16      8
            128  256   8       4
            256  512   4       2
        '''
        self.latent_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        Decoder_Module = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        for index in range(len(hidden_dims) - 1):
            Decoder_Module.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-index - 1],
                                       hidden_dims[-index - 2],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-index - 2]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*Decoder_Module)
        '''
             Cin  Cout  Sizein  Sizeout
             512  256   2       4
             256  128   4       8
             128  64    8       16
             64   32    16      32
             32   16    32      64
        '''
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0],
                               hidden_dims[0],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[0], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())
        '''
           Cin  Cout  Sizein  Sizeout
           16   16    64      128
           16   3     128     128
        '''

    def encoder_forward(self, original_img):
        result = self.encoder(original_img)
        result = self.encoder_flatten(result)

        mu = self.latent_mu(result)
        log_var = self.latent_log_var(result)

        return [mu, log_var]

    def reparameterize_latent_sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std)
        sample = std * z + mu
        return sample

    def decoder_forward(self, latent_sample):
        result = self.decoder_input(latent_sample)
        result = result.reshape(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, original_img):
        mu, log_var = self.encoder_forward(original_img)
        latent_sample = self.reparameterize_latent_sample(mu, log_var)
        reconstruct = self.decoder_forward(latent_sample)
        return [reconstruct, original_img, mu, log_var]

    def sample(self, sample_num, current_dev):
        z = torch.randn(sample_num, self.latent_dim)
        z = z.to(current_dev)
        samples = self.decoder_forward(z)
        return samples

    def generate(self, original_img, current_dev):
        original_img_dev = original_img.to(current_dev)
        reconstruction = self.forward(original_img_dev)[0]
        return reconstruction

    def loss_function(self, *args):
        reconstruct = args[0]
        original_img = args[1]
        mu = args[2]
        log_var = args[3]
        # Reconstruction Loss: MSE
        ReconLossFunction = torch.nn.MSELoss(reduction='mean')
        ReconLoss = ReconLossFunction(reconstruct, original_img)
        # 样本与重建图像的均值MSE损失
        # KL Divergence: Similarity of the latent space
        KLD_Loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1), dim=0)
        # 在latent维度上取和 再在样本维度上取均值
        Loss = ReconLoss + KLD_Loss  # 0.00032
        return Loss
