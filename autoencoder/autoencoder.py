import torch
import torch.nn as nn
import torch.nn.functional as F

class MushroomAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, image_size=256):
        super(MushroomAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        # Latent space
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid()  # Final output between 0 and 1
        )

    def encode(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(x.size(0), -1)
        mu = self.fc_mu(encoded)
        log_var = self.fc_logvar(encoded)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), 1024, 8, 8)
        return torch.clamp(self.decoder(z), 0.0, 1.0)  # Ensure output stays in [0, 1] range

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        reconstructed = self.decode(z)

        # Check for out-of-range values
        reconstructed_min, reconstructed_max = reconstructed.min().item(), reconstructed.max().item()
        if reconstructed_min < 0 or reconstructed_max > 1:
            print("Warning: Reconstructed image out of range before loss computation.")
            print(f"Reconstructed range: min {reconstructed_min}, max {reconstructed_max}")


        return reconstructed, mu, log_var
