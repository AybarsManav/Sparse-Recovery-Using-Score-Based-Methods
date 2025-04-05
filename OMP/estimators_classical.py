import sys
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

def omp_dct_estimator(A_val, y_batch_val, batch_size, eps=1e-3):
    # One can prove that taking 2D DCT of each row of A,
    # then solving using OMP, and finally taking 2D ICT gives the correct answer.
    A_new = copy.deepcopy(A_val)
    for i in range(A_val.shape[0]):
        A_new[i, :] = vec([dct2(channel) for channel in devec(A_new[i, :])])

    x_hat_batch = []
    for j in range(batch_size):
        y_val = y_batch_val[j]
        z_hat = orthogonal_matching_pursuit(A_new, y_val, eps)
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

def lasso_dct_estimator(A_val, y_batch_val, batch_size): 
    """LASSO with DCT"""
    # One can prove that taking 2D DCT of each row of A,
    # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
    A_new = copy.deepcopy(A_val)
    for i in range(A_val.shape[0]):
        A_new[i, :] = vec([dct2(channel) for channel in devec(A_new[i, :])])

    x_hat_batch = []
    for j in range(batch_size):
        y_val = y_batch_val[j]
        lasso_est = Lasso(alpha=1e-1, tol=1e-6, max_iter=10000, fit_intercept=False)
        lasso_est.fit(A_new, y_val)
        z_hat = lasso_est.coef_
        z_hat = np.reshape(z_hat, [-1])
        x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        x_hat_batch.append(x_hat)
    return x_hat_batch
    


if __name__ == "__main__":
    batch_size = 1
    dataset_path = "data/processed_celeba"  # Path to the folder with images
    # Initialize the CelebALoader
    celebA_loader = CelebALoader(root_dir=dataset_path, batch_size=batch_size, shuffle=False)
    # Get a batch of images
    dataloader = celebA_loader.get_dataloader()
    for images, _ in dataloader:  # `_` because we don't need labels
        images = images.numpy()
        break
    # Generate a random MxN measurement matrix A with elements sampled from gaussian 0 mean and 1/M variance
    M = 500 # Number measurements
    N = 64 * 64 * 3 # Number of pixels in each image
    A = np.random.randn(M, N)

    # Sample images using A
    y_batch = np.zeros((batch_size, M))
    for i in range(batch_size):
        y_batch[i] = np.matmul(A, images[i].reshape(-1))

    # Estimate images using lasso_dct_estimator
    # x_hat_batch = lasso_dct_estimator(A, y_batch, batch_size)
    # Estimate images using omp_dct_estimator
    x_hat_batch = omp_dct_estimator(A, y_batch, batch_size)

    # Display original and the estimated images
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow((images[0].transpose(1, 2, 0) + 1) / 2)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow((x_hat_batch[0].reshape(3, 64, 64).transpose(1, 2, 0) + 1) / 2)
    ax2.set_title("Estimated Image")
    ax2.axis('off')
    plt.show()





    