import einops
import imageio
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
import torch.nn as nn
import torch.nn.functional as F


# Code initially taken from the notebook for the practical class.
# ______________________________________________________________________________

# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(3, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + \
            (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

    # Update your generate_synthetic_samples method

    def generate_synthetic_samples(self, n_to_generate=10000, batch_size=128, n_steps=50, device=None):
        """Fast generation for evaluation"""
        self.eval()
        if device is None:
            device = self.device

        samples = []
        n_batches = (n_to_generate + batch_size - 1) // batch_size

        print(
            f"Generating {n_to_generate} samples in {n_batches} batches using {n_steps} steps...")

        for batch_idx in range(n_batches):
            current_batch_size = min(
                batch_size, n_to_generate - len(samples) * batch_size)

            with torch.no_grad():
                batch_samples = generate_new_images_fast(
                    self,
                    n_samples=current_batch_size,
                    n_steps=n_steps,  # Much faster!
                    device=device,
                    c=self.image_chw[0],
                    h=self.image_chw[1],
                    w=self.image_chw[2]
                )

                samples.append(batch_samples.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Generated {(batch_idx + 1) * batch_size} / {n_to_generate} samples")

        return torch.cat(samples, dim=0)[:n_to_generate]


def generate_new_images_fast(ddpm, n_samples=16, n_steps=50, device=None, c=3, h=28, w=28):
    """Fast generation using fewer steps"""
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h, w).to(device)

        # Use only every 20th step (1000/50 = 20)
        step_indices = torch.linspace(0, ddpm.n_steps-1, n_steps).int()

        print(f"Using steps: {step_indices.tolist()}")

        for i, t in enumerate(reversed(step_indices)):
            time_tensor = torch.ones(n_samples).to(device).long() * t
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) /
                                        (1 - alpha_t_bar).sqrt() * eta_theta)

            if i < len(step_indices) - 1:  # Not the last step
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

        x = torch.clamp(x, -1, 1)

    return (x + 1) * 0.5


def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(np.transpose(images[idx], (1, 2, 0)))
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


def generate_new_images(ddpm, n_samples=16, option=1, device=None, c=3, h=28, w=28):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # FIX: Create 1D time tensor, not 2D
            time_tensor = torch.ones(n_samples).to(
                device).long() * t  # Shape: (n_samples,)
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) /
                                        (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]

                if option == 1:
                    sigma_t = beta_t.sqrt()
                elif option == 2:
                    prev_alpha_t_bar = ddpm.alpha_bars[t -
                                                       1] if t > 0 else ddpm.alphas[0]
                    beta_tilda_t = ((1 - prev_alpha_t_bar) /
                                    (1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()
                else:
                    sigma_t = beta_t.sqrt()

                x = x + sigma_t * z

        x = torch.clamp(x, -1, 1)

    return (x + 1) * 0.5


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding



class FixedUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, in_channels=3):
        super(FixedUNet, self).__init__()
        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        self.in_channels = in_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(n_steps, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # Final layer 
        self.final = nn.Conv2d(64, in_channels, 1)

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        # Time embedding 
        batch_size = x.shape[0]
        t_emb = torch.zeros(batch_size, self.n_steps, device=x.device)
        t_emb.scatter_(1, t.unsqueeze(1), 1.0)
        t_emb = self.time_embed(t_emb)

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)

        # Bottleneck
        bottleneck = self.bottleneck(d2)

        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_conv2(u2)

        # Final 
        output = self.final(u2)

        return output
