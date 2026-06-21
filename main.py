import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QFrame, QMessageBox)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt, QTimer

class ThermalDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.init_ui()

    def init_ui(self):
        # تنظیمات پنجره اصلی
        self.setWindowTitle('سیستم پایش حرارتی و عیب‌یابی موتور')
        self.resize(800, 500)
        self.setStyleSheet("background-color: #f3f4f6; color: #1f2937;")

        # فونت اصلی
        main_font = QFont("Segoe UI", 10)
        self.setFont(main_font)

        # لی‌آوت اصلی (عمودی)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # عنوان برنامه
        title_label = QLabel('سیستم هوشمند عیب‌یابی موتور با پردازش تصویر حرارتی')
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #111827; margin: 10px;")
        main_layout.addWidget(title_label)

        # لی‌آوت میانی (افقی برای تقسیم صفحه به دو بخش)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # --- بخش راست: نمایش تصویر و آپلود ---
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: white; border-radius: 10px; border: 1px solid #d1d5db;")
        image_layout = QVBoxLayout()
        image_frame.setLayout(image_layout)

        self.image_label = QLabel('تصویر ترموگرام را انتخاب کنید')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #9ca3af; border: 2px dashed #d1d5db; border-radius: 5px;")
        self.image_label.setMinimumSize(350, 300)
        image_layout.addWidget(self.image_label)

        self.btn_upload = QPushButton('انتخاب تصویر حرارتی')
        self.btn_upload.setStyleSheet("background-color: #eff6ff; color: #1d4ed8; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.btn_upload.clicked.connect(self.upload_image)
        image_layout.addWidget(self.btn_upload)

        content_layout.addWidget(image_frame, stretch=1)

        # --- بخش چپ: نتایج تحلیل ---
        result_frame = QFrame()
        result_frame.setStyleSheet("background-color: white; border-radius: 10px; border: 1px solid #d1d5db;")
        result_layout = QVBoxLayout()
        result_frame.setLayout(result_layout)

        result_title = QLabel('نتایج پردازش CNN')
        result_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        result_layout.addWidget(result_title)

        self.lbl_status = QLabel('وضعیت: منتظر آپلود...')
        self.lbl_fault = QLabel('نوع عیب: -')
        self.lbl_confidence = QLabel('درصد اطمینان: -')

        for lbl in [self.lbl_status, self.lbl_fault, self.lbl_confidence]:
            lbl.setStyleSheet("padding: 5px; font-size: 14px;")
            result_layout.addWidget(lbl)

        result_layout.addStretch() # هل دادن محتوا به بالا

        self.btn_analyze = QPushButton('شروع پردازش و عیب‌یابی')
        self.btn_analyze.setStyleSheet("background-color: #2563eb; color: white; padding: 12px; border-radius: 5px; font-weight: bold;")
        self.btn_analyze.setEnabled(False) # غیرفعال تا زمانی که عکس آپلود شود
        self.btn_analyze.clicked.connect(self.analyze_image)
        result_layout.addWidget(self.btn_analyze)

        content_layout.addWidget(result_frame, stretch=1)

    # تابع هندل کردن آپلود تصویر
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'انتخاب تصویر', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            # تغییر اندازه عکس برای جا شدن در کادر با حفظ تناسب
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setStyleSheet("border: none;")
            
            # ریست کردن نتایج قبلی و فعال کردن دکمه پردازش
            self.lbl_status.setText('وضعیت: آماده پردازش')
            self.lbl_status.setStyleSheet("color: #1f2937;")
            self.lbl_fault.setText('نوع عیب: -')
            self.lbl_confidence.setText('درصد اطمینان: -')
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setStyleSheet("background-color: #2563eb; color: white; padding: 12px; border-radius: 5px; font-weight: bold;")

    # تابع شبیه‌سازی پردازش هوش مصنوعی
    def analyze_image(self):
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText('در حال پردازش (CNN)...')
        self.btn_analyze.setStyleSheet("background-color: #93c5fd; color: white; padding: 12px; border-radius: 5px;")
        
        # استفاده از تایمر برای شبیه‌سازی مکث ۱.۵ ثانیه‌ای پردازش مدل
        QTimer.singleShot(1500, self.show_results)

    def show_results(self):
        # اینجا بعداً مدل پایتونی شما (مثل model.predict) جایگزین می‌شود
        self.lbl_status.setText('وضعیت: هشدار (نیاز به بررسی)')
        self.lbl_status.setStyleSheet("color: #ea580c; font-weight: bold;") # رنگ نارنجی
        self.lbl_fault.setText('نوع عیب: خرابی بلبرینگ (Hotspot)')
        self.lbl_confidence.setText('درصد اطمینان: ۸۸.۵٪')
        
        self.btn_analyze.setText('شروع پردازش و عیب‌یابی')
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setStyleSheet("background-color: #2563eb; color: white; padding: 12px; border-radius: 5px; font-weight: bold;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # راست‌چین کردن کل برنامه برای زبان فارسی
    app.setLayoutDirection(Qt.RightToLeft)
    
    window = ThermalDashboard()
    window.show()
    sys.exit(app.exec())