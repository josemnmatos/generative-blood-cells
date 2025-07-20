# 🩸 Generating Blood Cells with ML 🧬

This repository contains the complete code and experiments for the paper **"Generating blood cells with ML"**. This project explores and compares three powerful generative models for synthesizing low-resolution (28x28) medical images of blood cells from the **BloodMNIST** dataset.

The goal is to see which model—a **CVAE**, **DCGAN**, or **DDPM**—is best suited for generating realistic and diverse images, a crucial task for data augmentation in the medical field where data is often scarce.

## 🚀 Results at a Glance

The experiments show that the **Denoising Diffusion Probabilistic Model (DDPM)** is the clear winner, producing higher-fidelity and more diverse samples than both the DCGAN and CVAE.

| Original Images | CVAE | DCGAN | DDPM (Best Model) |
| :---: | :---: | :---: | :---: |
| <img width="598" alt="Original Images" src="https://github.com/user-attachments/assets/345a83f1-2302-4f94-820b-0f5ab87c64a0" /> | <img width="628" alt="CVAE" src="https://github.com/user-attachments/assets/3fb0e3f7-ff4f-4baa-9f7b-c3f76b261261" /> | <img width="598" alt="DCGAN" src="https://github.com/user-attachments/assets/46ed67f1-492b-443d-8c43-f5b7de9bbc03" /> | <img width="598" alt="DDPM" src="https://github.com/user-attachments/assets/2b094827-9bc6-446d-960c-710aa214a750" /> |


The quality of the generated images was measured using the **Fréchet Inception Distance (FID)** score, where a lower score indicates better results. The final scores confirm the visual results:

| Model | Best FID Score (Mean ± Std over 5 runs) |
| :--- | :---: |
| Convolutional Variational Autoencoder (CVAE) | 170.05 ± 0.64 |
| Deep Convolutional Generative Adversarial Network (DCGAN) | 166.85 ± 0.56 |
| **Denoising Diffusion Probabilistic Model (DDPM)** | **122.26 ± 0.76** |

## 🧠 Models & Key Findings

### 1. Denoising Diffusion Probabilistic Model (DDPM)
The DDPM achieved the best performance by a large margin.
- **Finding:** Its success hinged on a long training schedule. Increasing the training from 100 to **500 epochs** was the single most important factor, dramatically reducing the FID score from ~202 to 122. This highlights that DDPMs benefit immensely from robust training to learn the intricate details of the data distribution.

### 2. Deep Convolutional Generative Adversarial Network (DCGAN)
This model produced sharp, high-quality images but was a challenge to train effectively.
- **Finding:** The training process was highly unstable. Stability was achieved by implementing **label smoothing** and using a **3:1 update ratio** for the generator and discriminator, which helped prevent the discriminator from overpowering the generator.

### 3. Convolutional Variational Autoencoder (CVAE)
The CVAE was the most stable model but fell short in image quality.
- **Finding:** Although easy to train, the CVAE produced noticeably "blurry" samples, a well-known limitation of VAEs in image generation tasks. Experimentation showed that a latent space dimension of 64 struck the best balance between capturing features and avoiding overfitting.

## 📂 Repository Structure

```
.
├── custom_models/
│   ├── cvae.py             # CVAE model architecture
│   ├── dcgan.py            # DCGAN Generator and Discriminator
│   └── ddpm.py             # U-Net for DDPM and helper functions
│
├── variational_autoencoder.ipynb  # Main notebook for CVAE experiments
├── dcgan.ipynb                    # Main notebook for DCGAN experiments
├── ddpm.ipynb                     # Main notebook for DDPM experiments
│
├── generative_pipeline.py  # A helper class to standardize training and evaluation
├── utils.py                # Utilities for loading the MedMNIST dataset
└── Generating blood cells with ML.pdf # The full research paper
```

## ⚡️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/josemnmatos/generative-blood-cells.git
cd generative-blood-cells
```

### 2. Set up the environment
It is recommended to use a virtual environment. The project relies on PyTorch and the MedMNIST library.
```bash
pip install torch torchvision torchaudio
pip install medmnist seaborn matplotlib numpy einops pytorch-fid-wrapper
```

### 3. Run the experiments
The main notebooks contain the complete workflow for training each model, generating samples, and evaluating the FID score.
```bash
# To run the CVAE model
jupyter notebook variational_autoencoder.ipynb

# To run the DCGAN model
jupyter notebook dcgan.ipynb

# To run the DDPM model
jupyter notebook ddpm.ipynb```
> **Note:** Training the DDPM for 500 epochs is computationally intensive and requires a GPU.
````

## 📄 Citation
If you find this work useful for your own projects or research, please consider citing the repo :)

**José Matos. "Generating blood cells with ML." (2025). Department of Informatics Engineering, University of Coimbra.**
