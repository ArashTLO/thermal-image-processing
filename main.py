import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QFrame, QGridLayout)
from PySide6.QtGui import QPixmap, QFont, QImage, QIcon
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# کلاس رسم نمودار (Matplotlib)
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
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('سیستم پایش حرارتی و عیب‌یابی صنعتی')
        self.resize(1200, 850)
        
        # --- استایل‌دهی کلی برنامه (QSS) با Hover و فونت ---
        self.setStyleSheet("""
            QWidget {
                font-family: 'Vazirmatn', 'IRANSans', 'Segoe UI', Tahoma;
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

        # --- بخش اول: کنترل‌ها و انتخاب پوشه ---
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(15, 10, 15, 10)
        
        self.btn_folder = QPushButton('📂 انتخاب پوشه تصاویر')
        self.btn_folder.clicked.connect(self.choose_folder)
        
        self.btn_prev = QPushButton('◀ تصویر قبلی')
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_image)
        
        self.btn_next = QPushButton('تصویر بعدی ▶')
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.next_image)

        self.lbl_info = QLabel('هیچ پوشه‌ای انتخاب نشده است')
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("color: #6b7280; font-style: italic;")

        control_layout.addWidget(self.btn_folder)
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_next)
        control_layout.addWidget(self.lbl_info, stretch=1)
        main_layout.addWidget(control_frame)

        # --- بخش دوم: نمایش مراحل تصویر (Pipeline) ---
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
        
        self.lbl_img_original = QLabel('تصویر اصلی')
        self.lbl_img_gray = QLabel('پیش‌پردازش')
        self.lbl_img_hotspot = QLabel('استخراج ویژگی')

        for lbl in [self.lbl_img_original, self.lbl_img_gray, self.lbl_img_hotspot]:
            lbl.setMinimumSize(350, 300)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(img_style)

        title_style = "font-weight: bold; font-size: 15px; color: #374151;"
        
        lbl_title_1 = QLabel("۱. تصویر ورودی")
        lbl_title_1.setStyleSheet(title_style)
        lbl_title_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title_2 = QLabel("۲. پیش‌پردازش (کاهش نویز)")
        lbl_title_2.setStyleSheet(title_style)
        lbl_title_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title_3 = QLabel("۳. تشخیص نقاط داغ")
        lbl_title_3.setStyleSheet(title_style)
        lbl_title_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        images_layout.addWidget(lbl_title_1, 0, 0)
        images_layout.addWidget(lbl_title_2, 0, 1)
        images_layout.addWidget(lbl_title_3, 0, 2)
        
        images_layout.addWidget(self.lbl_img_original, 1, 0)
        images_layout.addWidget(self.lbl_img_gray, 1, 1)
        images_layout.addWidget(self.lbl_img_hotspot, 1, 2)
        
        main_layout.addWidget(images_frame, stretch=2)

        # --- بخش سوم: نمودار و نتایج ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(15)
        
        # بوم رسم Matplotlib
        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        self.plot_canvas = MplCanvas(self, width=6, height=3, dpi=100)
        chart_layout.addWidget(self.plot_canvas)
        bottom_layout.addWidget(chart_frame, stretch=2)

        # پنل نتایج نهایی
        result_frame = QFrame()
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_result_title = QLabel('تحلیل تصویر فعلی:')
        lbl_result_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827;")
        
        self.lbl_status = QLabel('وضعیت: منتظر تصویر')
        self.lbl_status.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 5px; background: #f3f4f6;")
        
        self.lbl_max_temp = QLabel('بیشترین شدت حرارت: -')
        self.lbl_max_temp.setStyleSheet("font-size: 15px; color: #4b5563;")
        
        result_layout.addWidget(lbl_result_title)
        result_layout.addWidget(self.lbl_status)
        result_layout.addWidget(self.lbl_max_temp)
        result_layout.addStretch()
        
        bottom_layout.addWidget(result_frame, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=1)

    def choose_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "انتخاب پوشه تصاویر حرارتی")
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
                self.lbl_info.setText("هیچ تصویری در این پوشه یافت نشد!")
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
        self.lbl_info.setText(f"تصویر {self.current_index + 1} از {len(self.image_files)}: {os.path.basename(img_path)}")
        self.lbl_info.setStyleSheet("color: #10b981; font-weight: bold;")
        self.process_image_pipeline(img_path)

    def process_image_pipeline(self, img_path):
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

        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.hist(img_gray.ravel(), bins=256, range=(0, 256), color='#3b82f6', alpha=0.7)
        self.plot_canvas.axes.set_title('توزیع شدت پیکسلی (دما)', fontname='Tahoma', fontsize=10)
        
        # اصلاح آرگومان‌های تابع set_xlim
        self.plot_canvas.axes.set_xlim(0, 256)
        self.plot_canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.plot_canvas.draw()

        max_val = np.max(img_blurred)
        self.lbl_max_temp.setText(f'بیشترین شدت حرارت: {max_val} / 255')
        
        if max_val > 220:
            self.lbl_status.setText('وضعیت: 🔴 بحرانی')
            self.lbl_status.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 5px; background: #fee2e2; color: #991b1b; font-weight: bold;")
        elif max_val > 180:
            self.lbl_status.setText('وضعیت: 🟠 نیازمند بررسی')
            self.lbl_status.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 5px; background: #ffedd5; color: #9a3412; font-weight: bold;")
        else:
            self.lbl_status.setText('وضعیت: 🟢 عادی')
            self.lbl_status.setStyleSheet("font-size: 16px; padding: 10px; border-radius: 5px; background: #d1fae5; color: #065f46; font-weight: bold;")

    def display_cv_image(self, cv_img, label_widget, is_gray=False):
        h, w = cv_img.shape[:2]
        if is_gray:
            # اصلاح فرمت QImage برای تصاویر خاکستری
            q_img = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        else:
            bytes_per_line = 3 * w
            # اصلاح فرمت QImage برای تصاویر رنگی
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
    # اصلاح تراز راست‌چین برنامه
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    window = ThermalPipelineApp()
    window.show()
    sys.exit(app.exec())