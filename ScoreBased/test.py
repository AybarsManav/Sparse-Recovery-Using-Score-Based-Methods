import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, sys
sys.path.append('./')
from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.models.ema    import EMAHelper
from CelebA.celebA_loader import CelebALoader
import matplotlib.pyplot as plt
from torchvision import transforms
import time

metrics = False # Set to True to compute metrics and plot graphs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Is cuda available:", torch.cuda.is_available())

def computeNMSE(x, y):
    """
    Compute the Normalized Mean Squared Error (NMSE) between two tensors.
    Args:
        x: First tensor (estimated).
        y: Second tensor (ground truth).
    Returns:
        NMSE: Normalized Mean Squared Error.
    """
    return torch.norm(x - y) ** 2 / torch.norm(y) ** 2

def compute_noise_for_snr(signal, snr_db):
    """
    Compute the noise required to achieve a given SNR for a signal.
    
    Args:
        signal (torch.Tensor): The input signal (e.g., shape [batch_size, ...]).
        snr_db (float): Desired Signal-to-Noise Ratio in decibels (dB).
    
    Returns:
        noise (torch.Tensor): The noise tensor with the required power.
    """
    # Compute signal power
    signal_power = torch.mean(signal ** 2)
    
    # Compute noise power based on the desired SNR
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = signal_power / snr_linear
    
    # Generate noise with the required power
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    
    return noise, noise_power

def langevin_dynamics_reconstruction(model, current_estimate, images, A,
                        measurement_noise_power, measurement_noise,
                        alpha_step, beta_noise):
    """
    Perform Langevin dynamics for image reconstruction.

    Args:
        model: The score-based model.
        current_estimate: Current estimate of the image.
        images: Ground truth images.
        A: Measurement matrix.
        measurement_noise_power: Power of the noise during measurement.
        measurement_noise: Noise added to the measurements.
        alpha_step: Initial step size for Langevin dynamics.
        beta_noise: Grad noise scaling factor.

    Returns:
        Updated estimate after Langevin dynamics.
    """
    batch_size = current_estimate.shape[0]
    # Iterate through different noise levels for annealing
    for step_idx in tqdm(range(config.model.num_classes)):
        # Compute current step size and noise power for annealing
        current_sigma = model.sigmas[step_idx].item()
        labels = torch.ones(current_estimate.shape[0]).to(config.device) * step_idx
        labels = labels.long()

        # Compute annealing step size
        alpha = alpha_step * (current_sigma / config.model.sigma_end) ** 2

        # Iterate through Langevin steps (at a specific sigma)
        for langevin_step in range(config.test.langevin_steps):
            with torch.no_grad():
                score = model(current_estimate, labels)  # Compute the score

            # Add measurement noise through images
            noisy_images = images + measurement_noise

            # Conditional score
            Y = torch.matmul(A, noisy_images.view(noisy_images.shape[0], -1, 1))
            Y_hat = torch.matmul(A, current_estimate.view(current_estimate.shape[0], -1, 1))
            conditional_score = torch.matmul(A.T, Y - Y_hat) / (measurement_noise_power + current_sigma ** 2)

            # Sample noise
            grad_noise = np.sqrt(2 * alpha * beta_noise) * torch.randn_like(current_estimate)

            # Update the estimate
            current_estimate = current_estimate + \
                alpha * (score + conditional_score.view(batch_size, 3, 64, 64)) + grad_noise

    # Denoise
    with torch.no_grad():
        last_noise = (len(model.sigmas) - 1) * \
            torch.ones(current_estimate.shape[0], device=current_estimate.device)
        last_noise = last_noise.long()
        current_estimate = current_estimate + current_sigma ** 2 * model(current_estimate, last_noise)

    return current_estimate

# Load the model
target_file = "models/checkpoint_40000.pt"
contents = torch.load(target_file, weights_only=False)
config = contents['config']
config.device = device
model = NCSNv2Deepest(config=config)
model = model.to(config.device)

# If exponential moving average (EMA) is used, load the EMA state
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(model)
    ema_helper.load_state_dict(contents["ema_state"])
    ema_helper.ema(model)
else:
    model.load_state_dict(contents['model_state'])
model.eval()


N = 64 * 64 * 3 # Number of pixels in each image
M = 2500 # Number measurements
# Instead of random A matrix create an identity matrix and randomly select M rows
identity_matrix = np.eye(N)  # Create an identity matrix of size N x N
selected_rows = np.random.choice(N, M, replace=False)  # Randomly select M rows
A = identity_matrix[selected_rows, :]  # Select the rows from the identity matrix
A = torch.from_numpy(A).float()
A = A.to(config.device)

# Prepare the test dataset
dataset_path = "data/processed_celeba_test"  # Path to the folder with images
batch_size = 1 #maybe increase this when calculating metrics
celebA_loader = CelebALoader(root_dir=dataset_path, batch_size=batch_size, shuffle=False)
dataloader = celebA_loader.get_dataloader()

# Annealing parameters
alpha_step = 3.3e-6
beta_noise = 1.

# Testing SNR
snr_db = 1000  # Desired SNR in dB

# Test NMSE
NMSE = 0.0
num_samples = 0

# Test the model on the dataset
for images, _ in tqdm(dataloader):
    images = images.to(config.device)

    # Compute measurement noise and noise power for the given SNR.
    # Note that SNR is computed on images and noise matrix is sub_sampled through measurement
    measurement_noise, measurement_noise_power = compute_noise_for_snr(images, snr_db)

    # Generate random initial estimate
    current_estimate = torch.randn_like(images).to(config.device)

    # Langevin dynamics for reconstruction
    start_time = time.time()

    current_estimate = langevin_dynamics_reconstruction(
        model, current_estimate, images, A,
        measurement_noise_power, measurement_noise,
        alpha_step, beta_noise)

    end_time = time.time()
    print(f"Langevin dynamics reconstruction took {end_time - start_time:.2f} seconds.")

    # Compute NMSE for this batch
    NMSE += computeNMSE(current_estimate, images)
    num_samples += images.shape[0]

    break

print("NMSE:", NMSE.item() / num_samples)
# Save the generated images
save_dir = "data/saved"
os.makedirs(save_dir, exist_ok=True)
# Show a sample from the estimated images and original images
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(images[0].cpu().permute(1, 2, 0))
ax.set_title("Original Image")
ax.axis('off')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(current_estimate[0].cpu().permute(1, 2, 0))
ax.set_title("Estimated Image")
ax.axis('off')
plt.savefig(os.path.join(save_dir, "generated.png"))
plt.show()

# Save the first reconstructed image of the batch
reconstructed_image = current_estimate[0].cpu().permute(1, 2, 0).numpy()
reconstructed_image = np.clip(reconstructed_image, 0, 1)
plt.imsave(os.path.join(save_dir, "reconstructed.png"), reconstructed_image)
# Save the original image
original_image = images[0].cpu().permute(1, 2, 0).numpy()
original_image = np.clip(original_image, 0, 1)
plt.imsave(os.path.join(save_dir, "original.png"), original_image)

# This part is for generating plots and computing metrics
if metrics:
    ############### NMSE vs SNR for fixed M ################
    fixed_M = 2500
    snr_db_values = np.arange(-5, 25.1, 10)  # Different SNR(dB) values
    nmse_values_snr = []

    for snr_db in snr_db_values:
        NMSE = 0.0
        num_samples = 0

        for images, _ in tqdm(dataloader):
            images = images.to(config.device) # Get current batch of images

            current_estimate = torch.randn_like(images).to(config.device) # Initialize estimate

            # Compute measurement noise and noise power for the given SNR
            measurement_noise, noise_power = compute_noise_for_snr(images, snr_db)

            # Reconstruction process
            current_estimate = langevin_dynamics_reconstruction(
                model, current_estimate, images, A,
                noise_power, measurement_noise,
                alpha_step, beta_noise)

            NMSE += computeNMSE(current_estimate, images)
            num_samples += images.shape[0]
            break  # Only one batch for simplicity

        nmse_values_snr.append(NMSE.item() / num_samples)

        # Save the first reconstructed image of the batch
        reconstructed_image = current_estimate[0].cpu().permute(1, 2, 0).numpy()
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        plt.imsave(os.path.join(save_dir, f"reconstructed_snr_{snr_db}.png"), reconstructed_image)

        # Save the original image
        original_image = images[0].cpu().permute(1, 2, 0).numpy()
        original_image = np.clip(original_image, 0, 1)
        plt.imsave(os.path.join(save_dir, f"original_m_{M}.png"), original_image)

    ################# Different M for fixed SNR ################
    fixed_snr = 20  
    m_values = np.arange(500, 4501, 1000) # Different M values
    nmse_values_m = []

    for M in m_values:
        NMSE = 0.0
        num_samples = 0
        selected_rows = np.random.choice(N, M, replace=False)
        A = identity_matrix[selected_rows, :]
        A = torch.from_numpy(A).float().to(config.device)

        for images, _ in tqdm(dataloader):
            images = images.to(config.device)
            current_estimate = torch.randn_like(images).to(config.device) # Initialize estimate

            # Compute measurement noise and noise power for the given SNR
            measurement_noise, train_noise_power = compute_noise_for_snr(images, fixed_snr)

            # Reconstruction process
            current_estimate = langevin_dynamics_reconstruction(
                model, current_estimate, images, A,
                train_noise_power, measurement_noise,
                alpha_step, beta_noise)

            NMSE += computeNMSE(current_estimate, images)
            num_samples += images.shape[0]
            break  # Only one batch for simplicity

        nmse_values_m.append(NMSE.item() / num_samples)

        # Save the first reconstructed image of the batch
        reconstructed_image = current_estimate[0].cpu().permute(1, 2, 0).numpy()
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        plt.imsave(os.path.join(save_dir, f"reconstructed_m_{M}.png"), reconstructed_image)

    ########## Plot figures ##########
    # Plot NMSE vs SNR for fixed M
    plt.figure(figsize=(10, 5))
    plt.plot(snr_db_values, nmse_values_snr, marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE")
    plt.title("NMSE vs SNR (Fixed M = {})".format(fixed_M))
    plt.grid()
    plt.savefig(os.path.join(save_dir, "nmse vs snr.png"))
    plt.show()

    # Plot NMSE vs M for fixed SNR
    plt.figure(figsize=(10, 5))
    plt.plot(m_values, nmse_values_m, marker='o')
    plt.xlabel("Number of Measurements (M)")
    plt.ylabel("NMSE")
    plt.title("NMSE vs M (Fixed SNR = {})".format(fixed_snr))
    plt.grid()
    plt.savefig(os.path.join(save_dir, "nmse vs m.png"))
    plt.show()

    # Save NMSE values for further analysis
    np.save(os.path.join(save_dir, "m_values.npy"), m_values)
    np.save(os.path.join(save_dir, "nmse_values_m.npy"), nmse_values_m)

    np.save(os.path.join(save_dir, "nmse_values_snr.npy"), nmse_values_snr)
    np.save(os.path.join(save_dir, "snr_db_values.npy"), snr_db_values)
