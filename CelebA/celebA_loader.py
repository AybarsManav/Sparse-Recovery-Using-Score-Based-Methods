import os, sys
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
sys.path.append('./')

class CelebADataset(Dataset):
        def __init__(self, root_dir, transform=None):
            """
            Custom Dataset to handle flat directory of CelebA images.

            Args:
                root_dir (str): Path to the folder containing images.
                transform (callable, optional): Transformations to apply to the images.
            """
            self.root_dir = root_dir
            self.transform = transform
            self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB
            if self.transform:
                image = self.transform(image)
            return image, 0  # Return dummy label (0) since labels are not needed

class CelebALoader:
    def __init__(self, root_dir, batch_size=64, shuffle=True, crop_size=140, resize_size=64):
        """
        Initializes the CelebALoader class.

        Args:
            root_dir (str): Path to the folder containing CelebA images.
            batch_size (int): Number of images per batch.
            shuffle (bool): Whether to shuffle the dataset.
            crop_size (int): Size for center cropping the images.
            resize_size (int): Size to resize the cropped images.
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Define transformations for the images
        self.transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),     # Center crop to crop_size
            # transforms.Resize((resize_size, resize_size)),  # Resize cropped images
            transforms.ToTensor(),               # Convert images to PyTorch tensors
            # transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
        ])

        # Initialize the dataset
        self.dataset = CelebADataset(root_dir=self.root_dir, transform=self.transform)

    def get_dataloader(self):
        """
        Creates and returns a DataLoader for the CelebA dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the CelebA dataset.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)


# Example usage
if __name__ == "__main__":
    dataset_path = "data/img_align_celeba"  # Path to the folder with images
    batch_size = 64

    # Initialize the CelebALoader
    celebA_loader = CelebALoader(root_dir=dataset_path, batch_size=batch_size, shuffle=True)

    # Get the DataLoader
    dataloader = celebA_loader.get_dataloader()

    # Iterate through the DataLoader
    for images, _ in dataloader:  # `_` because we don't need labels
        print(images.shape)  # Example: torch.Size([64, 3, 64, 64])