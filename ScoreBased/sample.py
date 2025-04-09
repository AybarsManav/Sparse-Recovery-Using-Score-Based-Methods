import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, sys
from torchvision.utils import make_grid, save_image
sys.path.append('./')
from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.models.ema    import EMAHelper
from CelebA.celebA_loader import CelebALoader
import matplotlib.pyplot as plt
from ncsnv2.models import anneal_Langevin_dynamics

# Load the model
target_file = "models/checkpoint_25000.pt"
contents = torch.load(target_file, weights_only=False, map_location="cuda:0")
config = contents['config']
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

# Create initial noise
torch.manual_seed(42)  # Set seed for reproducibility
x_mod = torch.rand(10, config.data.channels,
                            config.data.image_size[0], config.data.image_size[1],
                            device=config.device)

# Using anneal Langevin Dynamics to sample from the model
images = anneal_Langevin_dynamics(x_mod, model, model.sigmas,
                             n_steps_each=config.sampling.n_steps_each,
                             step_lr=config.sampling.step_lr,
                             final_only=False, verbose=True, denoise=True)

sample = images[-1].view(images[-1].shape[0], config.data.channels,
                                    config.data.image_size[0],
                                    config.data.image_size[1])


image_grid = make_grid(sample, 5)
os.makedirs("samples/", exist_ok=True)
save_image(image_grid,
    os.path.join("samples/", 'image_grid_{}.png'.format(target_file.split('/')[-1].replace('.pt', ''))))