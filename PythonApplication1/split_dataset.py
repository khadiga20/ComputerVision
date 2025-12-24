import os
import shutil
import random

# Paths
dataset_path = r"C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")
train_path = os.path.join(images_path, "train")
val_path = os.path.join(images_path, "val")
train_labels = os.path.join(labels_path, "train")
val_labels = os.path.join(labels_path, "val")

# Create folders if not exist
for p in [train_path, val_path, train_labels, val_labels]:
    os.makedirs(p, exist_ok=True)

# List all images
all_images = [f for f in os.listdir(images_path) if f.endswith((".png",".jpg",".jpeg"))]

# Shuffle and split
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

# Move images and labels
for img in train_imgs:
    shutil.move(os.path.join(images_path, img), os.path.join(train_path, img))
    label_file = img.rsplit(".",1)[0] + ".txt"
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.move(os.path.join(labels_path, label_file), os.path.join(train_labels, label_file))

for img in val_imgs:
    shutil.move(os.path.join(images_path, img), os.path.join(val_path, img))
    label_file = img.rsplit(".",1)[0] + ".txt"
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.move(os.path.join(labels_path, label_file), os.path.join(val_labels, label_file))

print("Dataset split into train and val successfully!")
