#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
from sklearn.model_selection import train_test_split

dataset_path = "/kaggle/input/flickr2k/Flickr2K"

# Output path for processed data
output_path = "/kaggle/working/processed_data"
os.makedirs(output_path, exist_ok=True)


# In[2]:


image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".png")]

#ensure there are images to process
if len(image_files) == 0:
    raise ValueError("No images found in the dataset path.")

#split dataset
train_images, val_images = train_test_split(image_files, test_size = 0.2, random_state = 42)

print(f"Total images: {len(image_files)}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")


# ***Preprocess and save images*** 

# In[ ]:


def preprocess_and_save(image_paths, save_dir, downscale_factor=2):
    # Create directories for HR and LR images
    os.makedirs(os.path.join(save_dir, "HR"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "LR"), exist_ok=True)
    
    for image_path in image_paths:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load {image_path}, skipping.")
            continue

        # Save HR image
        hr_path = os.path.join(save_dir, "HR", os.path.basename(image_path))
        cv2.imwrite(hr_path, img)

        # Downsample (bicubic)
        height, width, _ = img.shape  # Handle colored images (height, width, channels)
        lr = cv2.resize(img, (width // downscale_factor, height // downscale_factor), interpolation=cv2.INTER_CUBIC)

        # Save LR image
        lr_path = os.path.join(save_dir, "LR", os.path.basename(image_path))
        cv2.imwrite(lr_path, lr)

# Process training and validation images
preprocess_and_save(train_images, os.path.join(output_path, "train"))
preprocess_and_save(val_images, os.path.join(output_path, "val"))

# Verify saved images
print("Processed training LR images:", len(os.listdir(os.path.join(output_path, "train/LR"))))
print("Processed training HR images:", len(os.listdir(os.path.join(output_path, "train/HR"))))
print("Processed validation LR images:", len(os.listdir(os.path.join(output_path, "val/LR"))))
print("Processed validation HR images:", len(os.listdir(os.path.join(output_path, "val/HR"))))