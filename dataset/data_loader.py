import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

DATASET_DIR = "./Dataset_Ready"

# Dictionary mapping class names to numeric labels (AI models only understand numbers)
CLASS_TO_IDX = {
    "Healthy": 0,
    "Cooling_Fault": 1,
    "Rotor_Fault": 2,
    "Stator_Fault": 3
}

class ThermalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse folders and collect all image paths and their labels
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Read the image using OpenCV
        image = cv2.imread(img_path)

        # Fix Pylance warning: ensure the image was loaded successfully
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted at:\n {img_path}")

        # Pylance now knows that the image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        # Apply transformations (resize and convert to PyTorch tensor)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Data Loader execution and testing section
if __name__ == "__main__":
    # Popular neural networks (such as ResNet) expect 224x224 images
    # ToTensor converts the image into a PyTorch tensor and normalizes pixel values to the range [0, 1]
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the complete dataset
    full_dataset = ThermalDataset(root_dir=DATASET_DIR, transform=transform_pipeline)
    print(f"✅ Total images found: {len(full_dataset)}")

    # Split the data: 80% for training and 20% for validation/testing
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders (batch_size=32 means 32 images are processed simultaneously)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"📦 Training samples: {len(train_dataset)} images")
    print(f"📦 Validation samples: {len(val_dataset)} images")

    # Test the output of a single batch
    images, labels = next(iter(train_loader))
    print(f"Output batch shape: {images.shape} -> (Batch, Channels, Height, Width)")
    print(f"Batch labels: {labels}")