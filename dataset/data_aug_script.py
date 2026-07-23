import cv2
import os
import random
import numpy as np

# Path to the dataset directory created in the previous step
DATASET_DIR = "./Dataset_Ready"

# Target number of images per class (balanced with the Stator class)
TARGET_COUNT = 300

# Classes that require data augmentation
CLASSES_TO_AUGMENT = ["Healthy", "Cooling_Fault", "Rotor_Fault"]

def augment_image(image):
    aug_img = image.copy()

    # 1. Horizontal flip (50% probability)
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)

    # 2. Small random rotation between -15 and +15 degrees
    angle = random.randint(-15, 15)
    h, w = aug_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Use BORDER_REPLICATE so that empty borders created during rotation
    # are filled with edge pixels instead of unnatural black borders
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Add very mild Gaussian noise (30% probability)
    if random.random() > 0.7:
        noise = np.random.normal(0, 3, aug_img.shape).astype(np.int16)

        # Prevent pixel value overflow (keep values within the range 0–255)
        aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return aug_img

def process_augmentation():
    for cls in CLASSES_TO_AUGMENT:
        cls_path = os.path.join(DATASET_DIR, cls)

        if not os.path.exists(cls_path):
            print(f"⚠️ Folder '{cls}' not found!")
            continue

        # List only the original images
        original_images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        current_count = len(original_images)

        if current_count == 0:
            print(f"⚠️ Folder '{cls}' is empty!")
            continue

        if current_count >= TARGET_COUNT:
            print(f"✅ Class '{cls}' already has enough images ({current_count} images).")
            continue

        print(f"🔄 Augmenting class '{cls}' (Current: {current_count} -> Target: {TARGET_COUNT})...")

        images_to_generate = TARGET_COUNT - current_count
        generated = 0

        while generated < images_to_generate:
            # Randomly select one of the original images
            random_img_name = random.choice(original_images)
            img_path = os.path.join(cls_path, random_img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Apply augmentation transformations
            augmented_img = augment_image(img)

            # Save the new image with the "aug_" prefix
            new_filename = f"aug_{generated}_{random_img_name}"
            cv2.imwrite(os.path.join(cls_path, new_filename), augmented_img)

            generated += 1

        print(f"✅ Class '{cls}' completed. ({generated} new images generated and added)")

if __name__ == "__main__":
    print("🚀 Starting dataset balancing process...")
    process_augmentation()
    print("🎉 All classes have been successfully balanced! The dataset is ready for training.")