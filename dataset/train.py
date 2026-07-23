import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import time
import copy

# Import custom classes created in previous steps
from data_loader import ThermalDataset, DATASET_DIR
from model import ThermalCNN

def train_model():
    # 1. Initial settings (Hyperparameters)
    BATCH_SIZE = 32
    EPOCHS = 15  # Number of times the model sees the entire dataset
    LEARNING_RATE = 0.001  # Learning step size (speed of weight updates)

    # Check for GPU availability to accelerate training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Processing on device: {device}")

    # 2. Data Preparation
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = ThermalDataset(root_dir=DATASET_DIR, transform=transform_pipeline)

    # Split data 80/20 for training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📦 Training data: {train_size} images")
    print(f"📦 Validation data: {val_size} images")

    # 3. Initialize model, loss function and optimizer
    model = ThermalCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("-" * 50)
    print("Starting neural network training process...")
    print("-" * 50)

    # 4. Main training loop (Epochs)
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # Each epoch contains two phases: Training and Validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode (enable Dropout)
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluation mode (disable Dropout)
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0  # Standard Python integer for Pylance compatibility

            # Iterate through image batches
            for inputs, labels in dataloader:
                # Move images and labels to GPU or CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset optimizer gradients at the beginning of each step
                optimizer.zero_grad()

                # Forward Pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # Find the class with the highest probability
                    loss = criterion(outputs, labels)

                    # Backward Pass and weight update only during training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Collect batch statistics
                running_loss += loss.item() * inputs.size(0)

                # Use .item() to extract value from PyTorch tensor
                running_corrects += torch.sum(preds == labels.data).item()

            # Select proper dataset size based on current phase
            current_size = train_size if phase == 'train' else val_size

            # Calculate epoch loss and accuracy using native Python values
            epoch_loss = running_loss / current_size
            epoch_acc = running_corrects / current_size

            print(f'{phase.capitalize()} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            # If validation phase achieves a new record, save the model weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print("-" * 50)
    print(f'🎉 Training completed in {time_elapsed // 60:.0f} minutes and {time_elapsed % 60:.0f} seconds')
    print(f'🏆 Best Validation Accuracy: {best_acc:.4f}')

    # 5. Load the best weights and save the final model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'thermal_motor_model.pth')
    print("💾 Best model saved as 'thermal_motor_model.pth'.")

if __name__ == '__main__':
    train_model()