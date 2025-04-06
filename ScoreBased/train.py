#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm

from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses        import get_optimizer
from ncsnv2.losses.dsm    import anneal_dsm_score_estimation

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dotmap           import DotMap
from CelebA.celebA_loader import CelebADataset
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='CDL-C')
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print("Is cuda available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config
config          = DotMap()
config.device   = device
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.num_classes   = 500 # 500
config.model.ngf           = 128 #128
config.model.sigma_begin = 90
config.model.sigma_end   = 0.01

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 1e-4
config.optim.beta1         = 0.90
config.optim.amsgrad       = False
config.optim.eps           = 3.3e-6
#the config uses 0.00000001

# Training
config.training.batch_size     = 128 #128 
config.training.num_workers    = 4  #nothing 
config.training.n_epochs       = 5 #500000
config.training.anneal_power   = 2 
config.training.log_all_sigmas = False

# Testing
config.test.langevin_steps = 5


# Data
config.data.channel        = args.train
config.data.channels       = 3 # all real
config.data.noise_std      = 0
config.data.image_size     = [64, 64] # 64
config.data.num_pilots     = config.data.image_size[1]
config.data.norm_channels  = 'global' #InstanceNorm++
config.data.logit_transform= False
config.data.rescaled       = True
#config.data.spacing_list   = [0.5] # Training and validation

transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Get datasets and loaders for channels
dataset = CelebADataset(root_dir='data/processed_celeba', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# # Choose the inference step size (epsilon) according to [Song '20]
# candidate_steps = np.logspace(-13, -8, 1000)
# step_criterion  = np.zeros((len(candidate_steps)))
# gamma_rate      = 1 / config.model.sigma_rate
# for idx, step in enumerate(candidate_steps):
#     step_criterion[idx] = (1 - step / config.model.sigma_end ** 2) \
#         ** (2 * config.model.num_classes) * (gamma_rate ** 2 -
#             2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
#                 1 - step / config.model.sigma_end ** 2) ** 2)) + \
#             2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
#                 1 - step / config.model.sigma_end ** 2) ** 2)
# best_idx = np.argmin(np.abs(step_criterion - 1.))
# config.model.step_size = candidate_steps[best_idx]

# Instantiate model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.to(config.device)
# Instantiate optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Instantiate counters and EMA helper
start_epoch, step = 0, 0
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# Get all sigma values for the discretized VE-SDE
sigmas = get_sigmas(config)

# Sample fixed validation data
#val_H_list = []
#for idx in range(len(config.data.spacing_list)):
#    val_sample = next(val_iters[idx])
#    val_H_list.append(val_sample['H_herm'].cuda())

# Logging
config.log_path = './models/'
os.makedirs(config.log_path, exist_ok=True)
train_loss, val_loss  = [], []

# For each epoch
for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    # For each batch
    for i, sample in tqdm(enumerate(train_loader)):
        diffuser.train()
        step += 1
        # Move data to device
        #for key in sample:
        #    sample[key] = sample[key].cuda()

        sample = sample[0]

        # Move data to device
        sample = sample.to(config.device)
        # Compute DSM loss using Hermitian channels
        loss = anneal_dsm_score_estimation(
            diffuser, sample, sigmas, None, 
            config.training.anneal_power)
        
        # Logging
        train_loss.append(loss.item())
        
        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update
        if config.model.ema:
            ema_helper.update(diffuser)
            
        # Verbose
        if step % 100 == 0 or step == 1:
            if config.model.ema:
                val_score = ema_helper.ema_copy(diffuser)
            else:
                val_score = diffuser
            
            # For each validation setup
            # Compute validation loss over the entire validation set
            local_val_losses = []
            diffuser.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for val_sample in val_loader:
                    # Move data to device
                    val_sample = val_sample[0].to(config.device)

                    # Compute DSM loss
                    val_dsm_loss = anneal_dsm_score_estimation(
                        val_score, val_sample, sigmas, None, config.training.anneal_power
                    ) 
                    local_val_losses.append(val_dsm_loss.item())
                    break # Only one batch for validation

                # Average validation loss over all batches
                average_val_loss = np.mean(local_val_losses)
                val_loss.append(average_val_loss)

        # Print validation loss
        print('Epoch %d, Step %d, Train Loss (EMA) %.3f, Val. Loss %.3f\n' % (
            epoch, step, loss, val_dsm_loss))
        
# Save final weights
torch.save({'model_state': diffuser.state_dict(),
            'optim_state': optimizer.state_dict(),
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss}, 
   os.path.join(config.log_path, 'final_model.pt'))