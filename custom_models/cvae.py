import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ConvEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(ConvEncoder, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4,
                      stride=2, padding=1),  # 32 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,
                      padding=1),  # 64 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,
                      padding=1),  # 128 x 16 x 16
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output
        )

        # fcc layers -> calc latent space distribution
        self.flattened_dim = 128 * 3 * 3

        self.dense_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.dense_logvar = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        self.flattened_dim = 128 * 3 * 3

        # fcc layers -> calc latent space distribution
        self.dense = nn.Linear(latent_dim, self.flattened_dim)

        # conv layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1),  # 64 x 6 x 6
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4,
                               stride=2, padding=1),  # 32 x 12 x 12
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels,
                               kernel_size=4, stride=2, padding=1),  # output_channels x 24 x 24
            nn.ReLU(),
            # Add one more layer to reach 28x28
            nn.ConvTranspose2d(output_channels, output_channels,
                               kernel_size=5, stride=1, padding=0),  # output_channels x 28 x 28
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.dense(z)
        # Reshape to (batch_size, channels, height, width)
        z = z.view(-1, 128, 3, 3)
        z = self.deconv(z)
        return z


class CVAE(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(CVAE, self).__init__()
        self.encoder = ConvEncoder(input_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, output_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

    def generate_synthetic_samples(self, n_to_generate, device, batch_size):
        synthetic_samples = []

        latent_vectors = torch.randn(
            n_to_generate, self.latent_dim).to(device)

        n_batches = (n_to_generate + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_to_generate)

            # Get a batch of latent vectors
            z_batch = latent_vectors[start_idx:end_idx]

            # Generate samples using the decoder
            samples = self.decoder(z_batch)

            # Reshape the flattened samples back
            channels, dim_1, dim_2 = 3, 28, 28
            reshaped_samples = samples.view(-1, channels, dim_1, dim_2)

            synthetic_samples.append(reshaped_samples.cpu())

        synthetic_samples = torch.cat(synthetic_samples, dim=0)

        return synthetic_samples


class VAELoss(nn.Module):
    def __init__(self, kld_weight=1.0):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recon_x, x, mu, logvar):
        batch_size = x.size(0)
        # Flatten both tensors to the same shape
        flattened_recon = recon_x.view(batch_size, -1)
        flattened_x = x.view(batch_size, -1)

        assert flattened_recon.shape == flattened_x.shape, f"Shape mismatch: {flattened_recon.shape} vs {flattened_x.shape}"
        assert torch.all(flattened_recon >= 0) and torch.all(
            flattened_recon <= 1), "recon_x values out of [0, 1] range"

        # Calculate binary cross entropy
        BCE = nn.functional.binary_cross_entropy(
            flattened_recon, flattened_x, reduction='sum'
        )

        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Return the total loss
        return BCE + self.kld_weight * KLD, BCE, KLD
