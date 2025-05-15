import torch
import torch.nn as nn
from torchvision import transforms


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

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

            # Reshape the flattened samples back to image format (if needed)
            # Assuming 3 channels and 28x28 image dimensions based on your dataset
            channels, dim_1, dim_2 = 3, 28, 28
            reshaped_samples = samples.view(-1, channels, dim_1, dim_2)

            synthetic_samples.append(reshaped_samples.cpu())

        # Concatenate all batches
        synthetic_samples = torch.cat(synthetic_samples, dim=0)

        data_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        # apply transform to samples

        synthetic_samples = data_transform(synthetic_samples)

        return synthetic_samples
