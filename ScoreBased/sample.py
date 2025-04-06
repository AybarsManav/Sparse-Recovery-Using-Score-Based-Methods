import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, sys
sys.path.append('./')
from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from CelebA.celebA_loader import CelebALoader
import matplotlib.pyplot as plt
from ncsnv2.models import anneal_Langevin_dynamics

# Load the model
target_file = "models/final_model.pt"
contents = torch.load(target_file, weights_only=False)
config = contents['config']
model = NCSNv2Deepest(config=config)
model = model.to(config.device)
model.load_state_dict(contents['model_state'])
model.eval()

# Create initial noise
x_mod = torch.randn(1, 3, 64, 64).to(config.device)

# Using anneal Langevin Dynamics to sample from the model
images = anneal_Langevin_dynamics(x_mod, model, model.sigmas, n_steps_each=5, step_lr=8e-6,
                             final_only=False, verbose=True, denoise=True)

# Plot the images
plt.figure(figsize=(10, 4))
selected_images = images[::50]  # Select every 50th image
for i, image in enumerate(selected_images[:10]):  # Ensure only 10 images are plotted
    plt.subplot(2, 5, i + 1)  # Create a 2x5 grid
    plt.imshow((image[0].permute(1, 2, 0).cpu().numpy() + 1) / 2)
    plt.axis('off')
plt.tight_layout()
plt.show()