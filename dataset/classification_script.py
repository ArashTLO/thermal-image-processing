import os
import shutil

# مسیر پوشه ای که همین الان دانلود و اکسترکت کردید
SOURCE_DIR = "آدرس_پوشه_اصلی_شما" # مثال: './raw_dataset'
TARGET_DIR = "./Dataset_Ready"

# نقشه کلاس‌بندی ما
CLASS_MAPPING = {
    "Healthy": ["Noload"],
    "Cooling_Fault": ["Fan"],
    "Rotor_Fault": ["Rotor-0"],
    "Stator_Fault": [
        "A10", "A30", "A50", 
        "A&C10", "A&C30", 
        "A&B50", 
        "A&C&B10", "A&C&B30"
    ]
}

def reorganize_dataset():
    # ساخت پوشه اصلی تارگت
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    for main_class, sub_folders in CLASS_MAPPING.items():
        # ساخت پوشه ۴ کلاس اصلی
        class_path = os.path.join(TARGET_DIR, main_class)
        os.makedirs(class_path, exist_ok=True)
        
        count = 0
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(SOURCE_DIR, sub_folder)
            
            if not os.path.exists(sub_folder_path):
                print(f"⚠️ هشدار: پوشه {sub_folder} پیدا نشد!")
                continue
                
            for file_name in os.listdir(sub_folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    src_file = os.path.join(sub_folder_path, file_name)
                    # تغییر نام فایل برای جلوگیری از جایگزینی (مثلاً: A10_1.jpg)
                    new_file_name = f"{sub_folder}_{file_name}"
                    dst_file = os.path.join(class_path, new_file_name)
                    
                    shutil.copy2(src_file, dst_file)
                    count += 1
                    
        print(f"✅ کلاس {main_class} ایجاد شد با {count} تصویر.")

if __name__ == "__main__":
    reorganize_dataset()
    print("🎉 ادغام دیتاست با موفقیت به پایان رسید!")