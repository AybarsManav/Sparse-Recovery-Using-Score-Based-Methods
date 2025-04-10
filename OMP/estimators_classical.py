import sys, os
sys.path.append("c:/dev/Sparse-Recovery-Using-Score-Based-Methods")
print(sys.path)
import copy
from numba import njit
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Lasso
from CelebA.celebA_loader import CelebALoader
import time

def compute_noise_for_snr(signal, snr_db):
    """
    Compute the noise required to achieve a given SNR for a signal.
    
    Args:
        signal (np.ndarray): The input signal (e.g., shape [batch_size, ...]).
        snr_db (float): Desired Signal-to-Noise Ratio in decibels (dB).
    
    Returns:
        noise (np.ndarray): The noise array with the required power.
        noise_power (float): The computed noise power.
    """
    # Compute signal power
    signal_power = np.mean(signal ** 2)
    
    # Compute noise power based on the desired SNR
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = signal_power / snr_linear
    
    # Generate noise with the required power
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    
    return noise, noise_power

def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')

def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')

def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])

def devec(vector):
    image = np.reshape(vector, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels

def omp_dct_estimator(Phi_us_T, y_batch_val, batch_size, eps=1e-3):
    # One can prove that taking 2D DCT of each row of PhiT,
    # then solving using OMP, and finally taking 2D ICT gives the correct answer.
    A = copy.deepcopy(Phi_us_T) # SubsampleMat * Phi.T * Psi which is measurement matrix A
    for i in range(A.shape[0]):
        A[i, :] = vec([dct2(channel) for channel in devec(A[i, :])])

    x_hat_batch = []
    for j in range(batch_size):
        y_val = y_batch_val[j]
        z_hat = orthogonal_matching_pursuit(A, y_val, eps)
        x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        x_hat_batch.append(x_hat)
    return x_hat_batch

@njit
def orthogonal_matching_pursuit(A, y, eps=1e-6):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for a single image measurement.
    Args:
        A: Measurement matrix.
        y: Measurements vector.
        eps: Tolerance for convergence.
    Returns:
        x_hat: Estimated sparse coefficients.
    """
   # Initialize variables
    M, N = A.shape  # Number of measurements and pixels
    residual = y.copy()
    indices = np.zeros(M, dtype=np.int32)  # Preallocate array for indices
    x_hat = np.zeros(N)
    num_selected = 0

    for i in range(M):
        # Find the index of the atom that best matches the residual
        correlations = np.abs(A.T @ residual)
        index = np.argmax(correlations)
        indices[num_selected] = index
        num_selected += 1

        # Update the solution using the selected atoms
        A_selected = A[:, indices[:num_selected]]  # Use only the selected indices
        x_hat_selected = np.linalg.pinv(A_selected) @ y
        x_hat[indices[:num_selected]] = x_hat_selected

        # Update the residual
        residual = y - A_selected @ x_hat_selected

        print("Iteration:", i + 1, "/", M, ", Residual norm:", np.linalg.norm(residual))

        # Check for convergence
        if np.linalg.norm(residual) < eps:
            break

    return x_hat

def lasso_dct_estimator(Phi_us_T, y_batch_val, batch_size): 
    """LASSO with DCT"""
    # One can prove that taking 2D DCT of each row of PhiT,
    # then solving using OMP, and finally taking 2D ICT gives the correct answer.
    A = copy.deepcopy(Phi_us_T) # SubsampleMat * Phi.T * Psi which is measurement matrix A
    for i in range(A.shape[0]):
        A[i, :] = vec([dct2(channel) for channel in devec(A[i, :])])

    x_hat_batch = []
    for j in range(batch_size):
        y_val = y_batch_val[j]
        lasso_est = Lasso(alpha=5e-6, tol=1e-6, max_iter=10000, fit_intercept=False)
        lasso_est.fit(A, y_val)
        z_hat = lasso_est.coef_
        z_hat = np.reshape(z_hat, [-1])
        x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        x_hat_batch.append(x_hat)
    return x_hat_batch
    
def compute_NMSE(image, x_hat):
    """Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(image.reshape(-1) - x_hat) ** 2 / np.linalg.norm(image.reshape(-1)) ** 2

if __name__ == "__main__":
    metrics = True # Set to True to compute metrics and generate plots
    save_dir = "plots_classical/" # Directory to save plots
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the dataset
    batch_size = 1
    dataset_path = "data/processed_celeba_test"  # Path to the folder with images
    celebA_loader = CelebALoader(root_dir=dataset_path, batch_size=batch_size, shuffle=False)

    # Get a batch of images
    dataloader = celebA_loader.get_dataloader()
    for images, _ in dataloader:  # `_` because we don't need labels
        images = images.permute(0, 2, 3, 1).numpy() * 2 - 1  # Normalize to [-1, 1]
        break

    # Generate a random MxN measurement matrix A with elements sampled from gaussian
    M = 10 # Number measurements
    N = 64 * 64 * 3 # Number of pixels in each image
    np.random.seed(42) # For reproducibility
    Phi = np.random.randn(N, N)  # Random Gaussian matrix
    Phi = Phi / np.linalg.norm(Phi, axis=0)  # Normalize columns of Phi to have unit l2 norm
    indices = np.random.choice(N, M, replace=False)  # Randomly select M indices from 0 to N-1
    Phi_us = Phi[:, indices]  # Select M columns from Phi - equivalent to selecting M rows from Phi.T

    # Sample images using Phi_us.T
    y_batch = np.zeros((batch_size, M))
    for i in range(batch_size):
        y_batch[i] = np.matmul(Phi_us.T, images[i].reshape(-1)) # Phi_us.T * s

    # Estimate images using lasso_dct_estimator
    # x_hat_batch = lasso_dct_estimator(Phi_us.T, y_batch, batch_size)
    # Estimate images using omp_dct_estimator
    x_hat_batch = omp_dct_estimator(Phi_us.T, y_batch, batch_size, 1e-6)

    # Compute NMSE
    NMSE = 0.0
    for i in range(batch_size):
        NMSE += compute_NMSE(images[i], x_hat_batch[i])
    NMSE /= batch_size
    print("NMSE:", NMSE)

    # Display original and the estimated images
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow((images[0] + 1) / 2)  # Rescale to [0, 1]
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow((x_hat_batch[0].reshape(64, 64, 3) + 1) / 2)  # Rescale to [0, 1]
    ax2.set_title("Estimated Image")
    ax2.axis('off')
    plt.show()

    # This part is for generating plots and computing metrics
    if metrics:
        ############### NMSE vs SNR for fixed M ################
        fixed_M = 2500
        snr_db_values = np.arange(-5, 25.1, 10)  # Different SNR(dB) values
        omp_nmse_values_snr = []
        lasso_nmse_values_snr = []

        for snr_db in snr_db_values:
            NMSE_lasso = 0.0
            NMSE_omp = 0.0
            num_samples = 0

            for images, _ in dataloader: # Take one batch of images
                images = images.permute(0, 2, 3, 1).numpy() * 2 - 1
                break

            # Compute measurement noise and noise power for the given SNR
            measurement_noise, noise_power = compute_noise_for_snr(images, snr_db)

            # Generate measurement matrix
            Phi = np.random.randn(N, N)  # Random Gaussian matrix
            Phi = Phi / np.linalg.norm(Phi, axis=0)  # Normalize columns of Phi to have unit l2 norm
            indices = np.random.choice(N, fixed_M, replace=False)  # Randomly select M indices from 0 to N-1
            Phi_us = Phi[:, indices]  # Select M columns from Phi - equivalent to selecting M rows from Phi.T
            
            # Add noise to the images
            images += measurement_noise

            # Sample the images
            y_batch = np.zeros((batch_size, fixed_M))
            for i in range(batch_size):
                y_batch[i] = np.matmul(Phi_us.T, images[i].reshape(-1))
            
            # Estimate using lasso and omp
            x_hat_batch_lasso = lasso_dct_estimator(Phi_us.T, y_batch, batch_size)
            x_hat_batch_omp = omp_dct_estimator(Phi_us.T, y_batch, batch_size, 1e-6)

            for i in range(batch_size):
                # Compute NMSE for each image in the batch
                NMSE_omp += compute_NMSE(images[i], x_hat_batch_omp[i])
                NMSE_lasso += compute_NMSE(images[i], x_hat_batch_lasso[i])
            num_samples += images.shape[0]

            omp_nmse_values_snr.append(NMSE_omp / num_samples)
            lasso_nmse_values_snr.append(NMSE_lasso / num_samples)

            # Save reconstructed OMP image
            reconstructed_image = x_hat_batch_omp[0].reshape(64, 64, 3)
            reconstructed_image = np.clip((reconstructed_image + 1) / 2, 0, 1)  # Rescale to [0, 1]
            plt.imsave(os.path.join(save_dir, f"omp_snr_{snr_db}.png"), reconstructed_image)

            # Save reconstructed Lasso image
            reconstructed_image = x_hat_batch_lasso[0].reshape(64, 64, 3)
            reconstructed_image = np.clip((reconstructed_image + 1) / 2, 0, 1)  # Rescale to [0, 1]
            plt.imsave(os.path.join(save_dir, f"lasso_snr_{snr_db}.png"), reconstructed_image)

        np.save(os.path.join(save_dir, "snr_db_values.npy"), snr_db_values)
        np.save(os.path.join(save_dir, "lasso_nmse_values_snr.npy"), lasso_nmse_values_snr)
        np.save(os.path.join(save_dir, "omp_nmse_values_snr.npy"), omp_nmse_values_snr)

        ################# Different M for fixed SNR ################
        fixed_snr = 20  # Fixed SNR value
        m_values = np.arange(500, 4501, 1000) # Different M values
        omp_nmse_values_m = []
        lasso_nmse_values_m = []
        lasso_times = []
        omp_times = []

        for M in m_values:
            NMSE_lasso = 0.0
            NMSE_omp = 0.0
            num_samples = 0

            for images, _ in dataloader: # Take one batch of images
                images = images.permute(0, 2, 3, 1).numpy() * 2 - 1
                break

            # Compute measurement noise and noise power for the given SNR
            measurement_noise, noise_power = compute_noise_for_snr(images, fixed_snr)

            # Generate measurement matrix
            Phi = np.random.randn(N, N)  # Random Gaussian matrix
            Phi = Phi / np.linalg.norm(Phi, axis=0)  # Normalize columns of Phi to have unit l2 norm
            indices = np.random.choice(N, M, replace=False)  # Randomly select M indices from 0 to N-1
            Phi_us = Phi[:, indices]  # Select M columns from Phi - equivalent to selecting M rows from Phi.T
            
            # Add noise to the images
            images += measurement_noise

            # Sample the images
            y_batch = np.zeros((batch_size, M))
            for i in range(batch_size):
                y_batch[i] = np.matmul(Phi_us.T, images[i].reshape(-1))
            
            # Estimate using lasso and omp

            start_time = time.time()
            x_hat_batch_lasso = lasso_dct_estimator(Phi_us.T, y_batch, batch_size)
            lasso_times.append(time.time() - start_time)

            if M < 3000:
                start_time = time.time()
                x_hat_batch_omp = omp_dct_estimator(Phi_us.T, y_batch, batch_size, 1e-6)
                omp_times.append(time.time() - start_time)

            for i in range(batch_size):
                # Compute NMSE for each image in the batch
                NMSE_omp += compute_NMSE(images[i], x_hat_batch_omp[i])
                NMSE_lasso += compute_NMSE(images[i], x_hat_batch_lasso[i])
            num_samples += images.shape[0]

            omp_nmse_values_m.append(NMSE_omp / num_samples)
            lasso_nmse_values_m.append(NMSE_lasso / num_samples)

            # Save reconstructed OMP image
            reconstructed_image = x_hat_batch_omp[0].reshape(64, 64, 3)
            reconstructed_image = np.clip(reconstructed_image / 2 + 0.5, 0, 1)
            plt.imsave(os.path.join(save_dir, f"omp_m_{M}.png"), reconstructed_image)
            
            # Save reconstructed Lasso image
            reconstructed_image = x_hat_batch_lasso[0].reshape(64, 64, 3)
            reconstructed_image = np.clip(reconstructed_image / 2 + 0.5, 0, 1)
            plt.imsave(os.path.join(save_dir, f"lasso_m_{M}.png"), reconstructed_image)


        # Save NMSE values for further analysis
        np.save(os.path.join(save_dir, "m_values.npy"), m_values)
        np.save(os.path.join(save_dir, "lasso_nmse_values_m.npy"), lasso_nmse_values_m)
        np.save(os.path.join(save_dir, "omp_nmse_values_m.npy"), omp_nmse_values_m)
        # Save timing results
        np.save(os.path.join(save_dir, "lasso_times.npy"), np.array(lasso_times))
        np.save(os.path.join(save_dir, "omp_times.npy"), np.array(omp_times))





    