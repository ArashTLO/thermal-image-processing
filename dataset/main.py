import sys
import os
import cv2
import numpy as np

# PySide6 Libraries
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QFrame, QGridLayout)
from PySide6.QtGui import QPixmap, QFont, QImage
from PySide6.QtCore import Qt

# Matplotlib Libraries
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# AI Libraries (PyTorch)
import torch
from torchvision import transforms
import torch.nn.functional as F

# Import our custom CNN architecture
from model import ThermalCNN

# Matplotlib Canvas Class
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.patch.set_facecolor('#ffffff')
        self.axes.set_facecolor('#ffffff')
        super().__init__(fig)

class ThermalPipelineApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_files = []
        self.current_index = -1
        
        # --- Initialize AI Model ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_model = None
        self.load_ai_model()
        
        self.init_ui()

    def load_ai_model(self):
        """Load the .pth file only once when the application starts"""
        try:
            print("⏳ Loading AI model...")
            self.ai_model = ThermalCNN(num_classes=4)
            
            # Find the exact path of the model file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'thermal_motor_model.pth')
            
            self.ai_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.ai_model.to(self.device)
            self.ai_model.eval() # Disable training mode (Dropout)
            print("✅ AI model loaded successfully and is ready to use.")
        except Exception as e:
            print(f"❌ Error loading AI model: {e}")
            self.ai_model = None

    def init_ui(self):
        self.setWindowTitle('Smart Thermal Monitoring & Motor Fault Diagnosis System')
        self.resize(1200, 850)
        
        # --- Global App Styling (QSS) ---
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Tahoma, Arial;
                font-size: 14px;
                background-color: #f3f4f6;
                color: #1f2937;
            }
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #e5e7eb;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                padding: 10px 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #e5e7eb;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Section 1: Controls & Folder Selection ---
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(15, 10, 15, 10)
        
        self.btn_folder = QPushButton('📂 Select Image Folder')
        self.btn_folder.clicked.connect(self.choose_folder)
        
        self.btn_prev = QPushButton('◀ Previous Image')
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_image)
        
        self.btn_next = QPushButton('Next Image ▶')
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.next_image)

        self.lbl_info = QLabel('No folder selected')
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("color: #6b7280; font-style: italic;")

        control_layout.addWidget(self.btn_folder)
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_next)
        control_layout.addWidget(self.lbl_info, stretch=1)
        main_layout.addWidget(control_frame)

        # --- Section 2: Image Pipeline Display ---
        images_frame = QFrame()
        images_layout = QGridLayout(images_frame)
        images_layout.setSpacing(15)
        images_layout.setContentsMargins(15, 15, 15, 15)
        
        img_style = """
            QLabel {
                background-color: #f9fafb; 
                border: 2px dashed #d1d5db; 
                border-radius: 8px;
            }
        """
        
        self.lbl_img_original = QLabel('Original Image')
        self.lbl_img_gray = QLabel('Preprocessing')
        self.lbl_img_hotspot = QLabel('Feature Extraction')

        for lbl in [self.lbl_img_original, self.lbl_img_gray, self.lbl_img_hotspot]:
            lbl.setMinimumSize(350, 300)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(img_style)

        title_style = "font-weight: bold; font-size: 15px; color: #374151;"
        
        lbl_title_1 = QLabel("1. Input Image")
        lbl_title_1.setStyleSheet(title_style)
        lbl_title_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title_2 = QLabel("2. Preprocessing (Noise Reduction)")
        lbl_title_2.setStyleSheet(title_style)
        lbl_title_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title_3 = QLabel("3. Hotspot Detection")
        lbl_title_3.setStyleSheet(title_style)
        lbl_title_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        images_layout.addWidget(lbl_title_1, 0, 0)
        images_layout.addWidget(lbl_title_2, 0, 1)
        images_layout.addWidget(lbl_title_3, 0, 2)
        
        images_layout.addWidget(self.lbl_img_original, 1, 0)
        images_layout.addWidget(self.lbl_img_gray, 1, 1)
        images_layout.addWidget(self.lbl_img_hotspot, 1, 2)
        
        main_layout.addWidget(images_frame, stretch=2)

        # --- Section 3: Chart & AI Results ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(15)
        
        # Matplotlib Plot Canvas
        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        self.plot_canvas = MplCanvas(self, width=6, height=3, dpi=100)
        chart_layout.addWidget(self.plot_canvas)
        bottom_layout.addWidget(chart_frame, stretch=2)

        # AI Result Panel
        result_frame = QFrame()
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_result_title = QLabel('AI Diagnosis Report:')
        lbl_result_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827;")
        
        # Main Status Label
        self.lbl_status = QLabel('System ready to receive image...')
        self.lbl_status.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 5px; background: #f3f4f6;")
        
        # AI Probabilities Details Label
        self.lbl_ai_details = QLabel('')
        self.lbl_ai_details.setStyleSheet("font-size: 13px; color: #4b5563; line-height: 1.5;")
        
        self.lbl_max_temp = QLabel('Max Thermal Intensity: -')
        self.lbl_max_temp.setStyleSheet("font-size: 14px; color: #6b7280; font-weight: bold;")
        
        result_layout.addWidget(lbl_result_title)
        result_layout.addWidget(self.lbl_status)
        result_layout.addWidget(self.lbl_max_temp)
        result_layout.addWidget(self.lbl_ai_details) 
        result_layout.addStretch()
        
        bottom_layout.addWidget(result_frame, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=1)

    def choose_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Thermal Images Folder")
        if folder_path:
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                                if f.lower().endswith(valid_extensions)]
            
            if self.image_files:
                self.current_index = 0
                self.update_ui_for_current_image()
                self.btn_next.setEnabled(len(self.image_files) > 1)
                self.btn_prev.setEnabled(False)
            else:
                self.lbl_info.setText("No images found in this folder!")
                self.lbl_info.setStyleSheet("color: #ef4444; font-weight: bold;")

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_ui_for_current_image()
            self.btn_prev.setEnabled(True)
            if self.current_index == len(self.image_files) - 1:
                self.btn_next.setEnabled(False)

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_ui_for_current_image()
            self.btn_next.setEnabled(True)
            if self.current_index == 0:
                self.btn_prev.setEnabled(False)

    def update_ui_for_current_image(self):
        img_path = self.image_files[self.current_index]
        self.lbl_info.setText(f"Image {self.current_index + 1} of {len(self.image_files)}: {os.path.basename(img_path)}")
        self.lbl_info.setStyleSheet("color: #10b981; font-weight: bold;")
        self.process_image_pipeline(img_path)

    def process_image_pipeline(self, img_path):

        # 1. Traditional OpenCV Processing
        img = cv2.imread(img_path)
        if img is None:
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_cv_image(img_rgb, self.lbl_img_original)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        self.display_cv_image(img_blurred, self.lbl_img_gray, is_gray=True)

        _, img_thresh = cv2.threshold(img_blurred, 200, 255, cv2.THRESH_BINARY)
        img_hotspot = img_rgb.copy()
        img_hotspot[img_thresh == 255] = [255, 0, 0] 
        self.display_cv_image(img_hotspot, self.lbl_img_hotspot)

        # Draw Histogram Plot
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.hist(img_gray.ravel(), bins=256, range=(0, 256), color='#3b82f6', alpha=0.7)
        self.plot_canvas.axes.set_title('Pixel Intensity Distribution (Temp)', fontname='Tahoma', fontsize=10)
        self.plot_canvas.axes.set_xlim(0, 256)
        self.plot_canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.plot_canvas.draw()

        max_val = np.max(img_blurred)
        self.lbl_max_temp.setText(f'Max thermal intensity in frame: {max_val} / 255')
        
        # 2. AI Processing (PyTorch Inference)
        if self.ai_model is not None:
            # Prepare image for AI model
            transform_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
            tensor_img = transform_pipeline(img_rgb)
            assert isinstance(tensor_img, torch.Tensor)
            input_tensor = tensor_img.unsqueeze(0).to(self.device)

            # Prediction
            with torch.no_grad():
                output = self.ai_model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                max_prob, predicted_class_idx = torch.max(probabilities, 0)
                
            class_idx = int(predicted_class_idx.item())
            confidence = float(max_prob.item()) * 100
            
            IDX_TO_CLASS = {
                0: "Healthy Motor",
                1: "Cooling System Fault",
                2: "Rotor Fault",
                3: "Stator Short Circuit"
            }
            predicted_name = IDX_TO_CLASS[class_idx]
            
            # Change color and icon based on AI diagnosis
            if class_idx == 0:
                # Green for Healthy Motor
                bg_color = "#d1fae5"
                text_color = "#065f46"
                icon = "🟢"
            else:
                # Red for any faults
                bg_color = "#fee2e2"
                text_color = "#991b1b"
                icon = "🔴"
                
            self.lbl_status.setText(f'{icon} Diagnosis: {predicted_name}\n🎯 Confidence: {confidence:.1f} %')
            self.lbl_status.setStyleSheet(f"font-size: 16px; padding: 15px; border-radius: 8px; background: {bg_color}; color: {text_color}; font-weight: bold;")
            
            # Print details of probabilities for other classes
            details_text = "Probabilities of other classes:\n"
            for idx, prob in enumerate(probabilities):
                if idx != class_idx:
                    details_text += f"▪ {IDX_TO_CLASS[idx]}: {float(prob.item()) * 100:.1f}%\n"
            self.lbl_ai_details.setText(details_text)
            
        else:
            self.lbl_status.setText('⚠️ Error: AI model is not loaded.')
            self.lbl_status.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 5px; background: #fef08a; color: #854d0e; font-weight: bold;")
            self.lbl_ai_details.setText("")

    def display_cv_image(self, cv_img, label_widget, is_gray=False):
        h, w = cv_img.shape[:2]
        if is_gray:
            q_img = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        else:
            bytes_per_line = 3 * w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setStyleSheet("background-color: white; border: 1px solid #d1d5db; border-radius: 8px;")
        
        label_widget.setPixmap(pixmap.scaled(
            label_widget.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    window = ThermalPipelineApp()
    window.show()
    sys.exit(app.exec())