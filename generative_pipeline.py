import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import copy
from custom_models.cvae import CVAE, VAELoss
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image

# **************************************
# Wrapper to use pytorch_fid's fid_score() with in-memory images
import pytorch_fid_wrapper as pfw


EVAL_SEEDS = [100, 200, 300, 400, 500]

ALLOWED_MODEL_CLASSES = [CVAE.__name__]

N_EVAL_SAMPLES = 10000


class CustomGenerativePipeline:
    def __init__(self, model: nn.Module,
                 criterion: nn.Module,
                 # e.g. {'lr': 0.001, 'weight_decay': 0}
                 optimizer_class: torch.optim.Optimizer,
                 optimizer_params: dict,
                 n_epochs: int,
                 train_dataloader: DataLoader,
                 train_dataset: ConcatDataset):
        pass
        self.eval_seeds = EVAL_SEEDS
        if model._get_name() not in ALLOWED_MODEL_CLASSES:
            return ValueError("Model class not allowed.")
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.n_epochs = n_epochs
        self.train_dataloader = train_dataloader
        self.train_dataset = train_dataset

        self.device = torch.device("cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu")

        print(f"Using device: {self.device}")

    def __init_pipeline(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)

        # Instantiate  optimizer for the current model
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_params
        )

        self._model_is_trained = False
        self.__training_losses = []
        self.__best_model_state = None
        self.__best_train_loss = np.inf
        self.__run_duration = None

    def train_model(self):
        """
        Train the model on the training data.
        """
        start_time = time.time()
        self.model.train()

        for epoch in range(self.n_epochs):
            training_loss = self.__training_loop()
            self.__training_losses.append(training_loss)

            print(
                f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {training_loss:.4f}")

            if training_loss < self.__best_train_loss:
                self.__best_train_loss = training_loss
                self.__best_model_state = copy.deepcopy(
                    self.model.state_dict())

        end_time = time.time()
        self.__run_duration = end_time - start_time

        # Revert to the best model state
        self.model.load_state_dict(self.__best_model_state)

    def __training_loop(self):
        """
        Epoch of model training.
        """
        batch_losses = []
        for X_batch, _ in self.train_dataloader:
            X_batch = X_batch.to(self.device)

            # Unpack model output for CVAE: (recon_x, mu, logvar)
            recon_x, mu, logvar = self.model(X_batch)
            loss, _, _ = self.criterion(recon_x, X_batch, mu, logvar)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())

        self._model_is_trained = True

        return np.mean(batch_losses)

    def __evaluate_model(self):
        """
        Evaluation loop for the model.
        """
        assert self._model_is_trained, "Model must be trained before evaluation."

        self.model.eval()


        fids = []

        for seed in self.eval_seeds:

            # 1. Set the seed

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # 2. Randomly sample N data points from the dataset

            random_sampler = RandomSampler(
                self.train_dataset, num_samples=N_EVAL_SAMPLES)

            eval_dataloader = DataLoader(
                dataset=self.train_dataset, batch_size=128, sampler=random_sampler)
            eval_tensor = torch.cat(
                [batch[0] for batch in eval_dataloader], dim=0)


            # 3. Generate N synthetic samples using the model

            synthetic_samples = self.model.generate_synthetic_samples(
                n_to_generate=N_EVAL_SAMPLES,
                device=self.device,
                batch_size=128
            )

            # 4. Compute Frechet Inception Distance (FID) between real and synthetic
            pfw.set_config(batch_size=10, dims=2048, device=self.device)

            print(f"Evaluating FID for seed {seed}...")
            # print(f"Real samples shape: {eval_tensor.shape}")
            # print(f"Fake samples shape: {synthetic_samples.shape}")

            fid = pfw.fid(fake_images=synthetic_samples,
                          real_images=eval_tensor,)

            print(f"FID:{fid}")

            fids.append(fid)

        # 5. Report average and std over seeded runs
        self.__fid_mean = np.mean(fids)
        self.__fid_std = np.std(fids)
        self.__fids = fids

        print(f"FID = {self.__fid_mean:.4f} +- {self.__fid_std:.4f}")

    def execute(self):
        """
        Trains the model.
        """
        self.__init_pipeline()
        self.train_model()

    def evaluate(self):
        """
        Evaluates the model.
        """
        self.__evaluate_model()

    def visualize(self, n_samples=10):
        """
        Visualizes original samples, their reconstructions, and generated samples.

        Args:
            n_samples (int): Number of samples to visualize for each category.
        """
        assert self._model_is_trained, "Model must be trained before visualization."

        # Get original samples from the dataset
        dataiter = iter(self.train_dataloader)
        images, _ = next(dataiter)
        original_samples = images[:n_samples].to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed_samples = self.model(original_samples)[0]

        # Generate synthetic samples
        synthetic_samples = self.model.generate_synthetic_samples(
            n_to_generate=n_samples,
            device=self.device,
            batch_size=n_samples
        )

        # Reshape synthetic samples
        synthetic_samples = synthetic_samples.view(n_samples, 3, 28, 28)

        # Plot the samples in three rows
        plt.figure(figsize=(15, 10))

        # Original samples
        plt.subplot(3, 1, 1)
        grid = make_grid(original_samples, nrow=5, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Original Samples")
        plt.axis('off')

        # Reconstructed samples
        plt.subplot(3, 1, 2)
        grid = make_grid(reconstructed_samples, nrow=5, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Reconstructed Samples")
        plt.axis('off')

        # Generated samples
        plt.subplot(3, 1, 3)
        grid = make_grid(synthetic_samples, nrow=5, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Generated Samples")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def get_losses(self):
        """Returns the training and validation losses per epoch."""
        return {"train": self.__training_losses}

    def get_run_duration(self):
        """Returns the duration of the last training run in seconds."""
        return self.__run_duration
