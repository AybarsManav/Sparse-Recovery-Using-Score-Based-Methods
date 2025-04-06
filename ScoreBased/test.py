import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, sys
sys.path.append('./')
from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from CelebA.celebA_loader import CelebALoader
import matplotlib.pyplot as plt

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

# Load the model
target_file = "models/final_model.pt"
contents = torch.load(target_file, weights_only=False)
config = contents['config']
model = NCSNv2Deepest(config=config)
model = model.to(config.device)
model.load_state_dict(contents['model_state'])
model.eval()

# The A should be the same with the one used in OMP or LASSO
M = 5000 # Number measurements
N = 64 * 64 * 3 # Number of pixels in each image
# A = np.random.randn(M, N) / np.sqrt(M)  # Normalized to have variance 1/M
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
batch_size = 10
celebA_loader = CelebALoader(root_dir=dataset_path, batch_size=batch_size, shuffle=False)
dataloader = celebA_loader.get_dataloader()

# Annealling parameters
alpha_step = 3e-11
beta_noise = 0.01

# Test NMSE
NMSE = 0.0
num_samples = 0

# Test the model on the dataset
for images, _ in tqdm(dataloader):
    images = images.to(config.device)

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
            Y = (A @ images.view(images.shape[0], -1).T).T
            Y_hat = (A @ current_estimate.view(current_estimate.shape[0], -1).T).T
            conditional_score = (A.T @ (Y - Y_hat).T).T / (train_noise_power + current_sigma ** 2)

            # Sample noise
            grad_noise = np.sqrt(2 * alpha * beta_noise) * torch.randn_like(current_estimate)

            # Update the estimate
            current_estimate = current_estimate \
            + alpha * (score - conditional_score.view(batch_size, 3, 64, 64)) + grad_noise
    # Denoise
    with torch.no_grad():
        last_noise = (len(model.sigmas) - 1) * \
              torch.ones(current_estimate.shape[0], device=current_estimate.device)
        last_noise = last_noise.long()
        current_estimate = current_estimate + current_sigma ** 2 * model(current_estimate, last_noise)

    # Compute NMSE for this batch
    NMSE += computeNMSE(current_estimate, images)
    num_samples += images.shape[0]

print("NMSE:", NMSE.item() / num_samples)

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
plt.show()