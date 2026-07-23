import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# initiate classification
from data_loader import ThermalDataset, DATASET_DIR
from model import ThermalCNN

def evaluate_and_plot():
    # Initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"evaluating model on device: {device}")

    class_names = ['Healthy', 'Cooling Fault', 'Rotor Fault', 'Stator Fault']

    # Reloading data
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    full_dataset = ThermalDataset(root_dir=DATASET_DIR, transform=transform_pipeline)
    
    # Testing model performance on whole dataset
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

    # Loading trained model
    model = ThermalCNN(num_classes=4)
    try:
        model.load_state_dict(torch.load('thermal_motor_model.pth', map_location=device))
    except FileNotFoundError:
        print("❌ thermal_motor_model.pth not found! First train the model.")
        return
        
    model.to(device)
    model.eval() # Dropout training layers

    # Predicted and true values lists
    y_true = []
    y_pred = []

    print("🔍 Extracting model predictions on images...")

    # Extracting predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Transfering data (Tensor to py lists)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calculating confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Full text report on terminal
    print("\n" + "="*50)
    print("📊 Classification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Heatmap matrix
    plt.figure(figsize=(8, 6))
    
    # Graphical setting diagram
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Thermal Motor Fault - Confusion Matrix', fontsize=16, pad=15)
    plt.ylabel('Actual Fault', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Fault', fontsize=12, labelpad=10)
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    print("✅ Matrix was drawn! check the diagram window.")
    plt.show()

if __name__ == '__main__':
    evaluate_and_plot()