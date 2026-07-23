import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ThermalCNN, self).__init__()

        # First layer: receives an RGB image (3 channels) and extracts 16 feature maps
        # Input: [Batch, 3, 224, 224] -> Output after max pooling: [Batch, 16, 112, 112]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second layer: extracts more complex thermal patterns
        # Output after max pooling: [Batch, 32, 56, 56]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third layer: deepest layer for final fault localization
        # Output after max pooling: [Batch, 64, 28, 28]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the 3D feature maps into a 1D vector
        # Dimensions: 64 channels × 28 height × 28 width = 50,176 features
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)  # Helps prevent overfitting

        # Final output layer: classifies the input into 4 classes
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass data through convolutional layers + ReLU activation + pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten the feature maps before the fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ---------------------------------------------------------
# Model integrity test (without training)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Create an instance of the model
    model = ThermalCNN(num_classes=4)
    print("Model architecture:\n", model)

    # Simulate a batch of 8 blank images with a resolution of 224×224
    dummy_input = torch.randn(8, 3, 224, 224)

    # Pass the simulated images through the model
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} -> (8 images, 4 class probabilities)")