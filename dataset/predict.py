import torch
import cv2
from torchvision import transforms
import torch.nn.functional as F
import os

# Import the model architecture that we created
from model import ThermalCNN

def predict_image(image_path, model_weight_path):
    # 1. Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at path {image_path}!")
        return
    if not os.path.exists(model_weight_path):
        print(f"❌ Error: Model file not found at path {model_weight_path}!")
        return

    # 2. Class dictionary (reverse mapping compared to training)
    IDX_TO_CLASS = {
        0: "Healthy",
        1: "Cooling Fault",
        2: "Rotor Fault",
        3: "Stator Fault"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize the model and load trained weights (.pth)
    model = ThermalCNN(num_classes=4)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    model.eval()

    # 4. Image preprocessing
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Image file is corrupted or cannot be read by OpenCV.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Apply transformations to the image
    tensor_img = transform_pipeline(image_rgb)

    # Fix Pylance warning: ensure the output is definitely a PyTorch tensor
    assert isinstance(tensor_img, torch.Tensor), "Error converting image to tensor"

    # Add batch dimension
    input_tensor = tensor_img.unsqueeze(0).to(device)

    print("🔍 Analyzing thermal image...")

    # 5. Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        max_prob, predicted_class_idx = torch.max(probabilities, 0)

    # Fix Pylance warning: explicitly convert index to a standard Python integer
    class_idx = int(predicted_class_idx.item())

    # 6. Display results
    predicted_name = IDX_TO_CLASS[class_idx]
    confidence = float(max_prob.item()) * 100

    print("-" * 50)
    print(f"✅ Detection Result: {predicted_name}")
    print(f"📊 AI Confidence Level: {confidence:.2f}%")
    print("-" * 50)

    print("Probability details:")
    for idx, prob in enumerate(probabilities):
        print(f" - {IDX_TO_CLASS[idx]}: {float(prob.item()) * 100:.2f}%")

if __name__ == "__main__":
    MODEL_PATH = "thermal_motor_model.pth"
    # Set the path of the image you want to test here:
    TEST_IMAGE_PATH = "./Dataset_Ready/Healthy/Noload_1.jpg"

    predict_image(TEST_IMAGE_PATH, MODEL_PATH)