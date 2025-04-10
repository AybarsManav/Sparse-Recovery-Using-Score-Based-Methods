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


metrics = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def compute_snr(signal, noise_power):

    signal_power = torch.norm(signal) ** 2 / signal.numel()
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

# Load the model
target_file = "models/final_model_1_epoch.pt"
contents = torch.load(target_file, weights_only=False)
config = contents['config']
config.device = device
model = NCSNv2Deepest(config=config)
model = model.to(config.device)

if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(model)
    ema_helper.load_state_dict(contents["ema_state"])
    ema_helper.ema(model)
else:
    model.load_state_dict(contents['model_state'])

model.eval()
print("Is cuda available:", torch.cuda.is_available())
# The A should be the same with the one used in OMP or LASSO
N = 64 * 64 * 3 # Number of pixels in each image
# M = np.astype(np.floor(N / 2), np.int32) # Number measurements
M = 2500 # Number measurements
# A = np.random.randn(M, N) # Normalized to have variance 1/M
# Instead of random A matrix create an identity matrix and randomly select M rows
identity_matrix = np.eye(N)  # Create an identity matrix of size N x N
selected_rows = np.random.choice(N, M, replace=False)  # Randomly select M rows
A = identity_matrix[selected_rows, :]  # Select the rows from the identity matrix
# A = contents['A'] # Get the measurement matrix A
# [M , N] = A.shape # Get the dimensions of A
A = torch.from_numpy(A).float()
A = A.to(config.device)

train_noise_power = 0.01 # Default noise power
# train_noise_power = contents['train_noise_power'] # Get the noise power
train_noise_power = torch.tensor(train_noise_power).float()
train_noise_power = train_noise_power.to(config.device)

# Prepare the test dataset
dataset_path = "data/processed_celeba_test"  # Path to the folder with images
batch_size = 1 #maybe increase this when calculating metrics

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5)  # Always flip the image
])

celebA_loader = CelebALoader(root_dir=dataset_path, batch_size=batch_size, shuffle=False, transform=transform)
dataloader = celebA_loader.get_dataloader()



# Annealling parameters
alpha_step = 3.3e-6
beta_noise = 1.

# Test NMSE
NMSE = 0.0
num_samples = 0

# Test the model on the dataset
for images, _ in tqdm(dataloader):

    #this is very hardcoded sorry aybars if u see this<3
    images = images.to(config.device)
    #images = torch.flip(images, [2])
    # Generate random initial estimate
    current_estimate = torch.randn_like(images).to(config.device)

    # Iterate through different noise levels
    for step_idx in tqdm(range(config.model.num_classes)):
        # Compute current step size and noise power
        current_sigma = model.sigmas[step_idx].item()
        labels = torch.ones(current_estimate.shape[0]).to(config.device) * step_idx
        labels = labels.long()

        # Compute annealed step size
        alpha = alpha_step * (current_sigma / config.model.sigma_end) ** 2

        for langevin_step in range(config.test.langevin_steps):
            with torch.no_grad():
                score = model(current_estimate, labels) # Compute the score
            
            # Conditional score
            Y = torch.matmul(A, images.view(images.shape[0], -1, 1))
            Y_hat = torch.matmul(A, current_estimate.view(current_estimate.shape[0], -1, 1))
            conditional_score = torch.matmul(A.T, Y - Y_hat) / (train_noise_power + current_sigma ** 2)

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

    # Compute NMSE for this batch
    NMSE += computeNMSE(current_estimate, images)
    num_samples += images.shape[0]

    break

print("NMSE:", NMSE.item() / num_samples)


save_dir = "data/saved"
os.makedirs(save_dir, exist_ok=True)


# Show a sample from the estimated images and original images
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow((images[0].cpu().permute(1, 2, 0) + 1) / 2)
ax.set_title("Original Image")
ax.axis('off')
ax = fig.add_subplot(1, 2, 2)
ax.imshow((current_estimate[0].cpu().permute(1, 2, 0) + 1) / 2)
ax.set_title("Estimated Image")
ax.axis('off')

plt.savefig(os.path.join(save_dir, "generated.png"))
plt.show()



if metrics:
    metrics_file = os.path.join(save_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        # NMSE vs SNR for fixed M
        fixed_M = 2500
        snr_values = []
        nmse_values_snr = []
        noise_powers = [0.001, 0.01, 0.1, 1.0]  # Different noise power levels

        f.write("NMSE vs SNR (Fixed M):\n")
        for noise_power in noise_powers:
            NMSE = 0.0
            num_samples = 0
            train_noise_power = torch.tensor(noise_power).float().to(config.device)

            for images, _ in tqdm(dataloader):
                images = images.to(config.device)
                current_estimate = torch.randn_like(images).to(config.device)

                # Reconstruction process (same as before)
                for step_idx in range(config.model.num_classes):
                    current_sigma = model.sigmas[step_idx].item()
                    labels = torch.ones(current_estimate.shape[0]).to(config.device) * step_idx
                    labels = labels.long()
                    alpha = alpha_step * (current_sigma / config.model.sigma_end) ** 2

                    for langevin_step in range(config.test.langevin_steps):
                        with torch.no_grad():
                            score = model(current_estimate, labels)
                        Y = torch.matmul(A, images.view(images.shape[0], -1, 1))
                        Y_hat = torch.matmul(A, current_estimate.view(current_estimate.shape[0], -1, 1))
                        conditional_score = torch.matmul(A.T, Y - Y_hat) / (train_noise_power + current_sigma ** 2)
                        grad_noise = np.sqrt(2 * alpha * beta_noise) * torch.randn_like(current_estimate)
                        current_estimate = current_estimate + alpha * (score + conditional_score.view(batch_size, 3, 64, 64)) + grad_noise

                with torch.no_grad():
                    last_noise = (len(model.sigmas) - 1) * torch.ones(current_estimate.shape[0], device=current_estimate.device).long()
                    current_estimate = current_estimate + current_sigma ** 2 * model(current_estimate, last_noise)

                NMSE += computeNMSE(current_estimate, images)
                num_samples += images.shape[0]
                break  # Only one batch for simplicity

            snr = compute_snr(images, noise_power)
            snr_values.append(snr)
            nmse = NMSE.item() / num_samples
            nmse_values_snr.append(nmse)

            f.write(f"SNR: {snr:.2f} dB, NMSE: {nmse:.6f}\n")

        # NMSE vs M (Fixed SNR)
        fixed_snr = 20  
        m_values = [1000, 2000, 3000, 4000, 5000]  
        nmse_values_m = []

        f.write("\nNMSE vs M (Fixed SNR):\n")
        for M in m_values:
            NMSE = 0.0
            num_samples = 0
            selected_rows = np.random.choice(N, M, replace=False)
            A = identity_matrix[selected_rows, :]
            A = torch.from_numpy(A).float().to(config.device)

            for images, _ in tqdm(dataloader):
                images = images.to(config.device)
                current_estimate = torch.randn_like(images).to(config.device)

                for step_idx in range(config.model.num_classes):
                    current_sigma = model.sigmas[step_idx].item()
                    labels = torch.ones(current_estimate.shape[0]).to(config.device) * step_idx
                    labels = labels.long()
                    alpha = alpha_step * (current_sigma / config.model.sigma_end) ** 2

                    for langevin_step in range(config.test.langevin_steps):
                        with torch.no_grad():
                            score = model(current_estimate, labels)
                        Y = torch.matmul(A, images.view(images.shape[0], -1, 1))
                        Y_hat = torch.matmul(A, current_estimate.view(current_estimate.shape[0], -1, 1))
                        conditional_score = torch.matmul(A.T, Y - Y_hat) / (train_noise_power + current_sigma ** 2)
                        grad_noise = np.sqrt(2 * alpha * beta_noise) * torch.randn_like(current_estimate)
                        current_estimate = current_estimate + alpha * (score + conditional_score.view(batch_size, 3, 64, 64)) + grad_noise

                with torch.no_grad():
                    last_noise = (len(model.sigmas) - 1) * torch.ones(current_estimate.shape[0], device=current_estimate.device).long()
                    current_estimate = current_estimate + current_sigma ** 2 * model(current_estimate, last_noise)

                NMSE += computeNMSE(current_estimate, images)
                num_samples += images.shape[0]
                break  # Only one batch for simplicity

            nmse = NMSE.item() / num_samples
            nmse_values_m.append(nmse)

            f.write(f"M: {M}, NMSE: {nmse:.6f}\n")

        
    plt.figure(figsize=(10, 5))
    plt.plot(snr_values, nmse_values_snr, marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE")
    plt.title("NMSE vs SNR (Fixed M)")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "nmse vs snr.png"))

    plt.show()



    plt.figure(figsize=(10, 5))
    plt.plot(m_values, nmse_values_m, marker='o')
    plt.xlabel("Number of Measurements (M)")
    plt.ylabel("NMSE")
    plt.title("NMSE vs M (Fixed SNR)")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "nmse vs m.png"))

    plt.show()
