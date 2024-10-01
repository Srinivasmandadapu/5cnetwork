import cv2
import numpy as np
import os
from tifffile import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image_and_mask(patient_folder):
    images = []
    masks = []
    
    for file in os.listdir(patient_folder):
        if file.endswith('.tif') and '_mask' not in file:
            # Load the image
            image = imread(os.path.join(patient_folder, file))
            
            # Handle missing sequences by replacing them with FLAIR
            if image.shape[2] == 3:
                pre_contrast = image[:, :, 0]
                flair = image[:, :, 1]
                post_contrast = image[:, :, 2]
                
                if np.array_equal(pre_contrast, flair):
                    pre_contrast = flair
                if np.array_equal(post_contrast, flair):
                    post_contrast = flair
                
                images.append(np.stack([pre_contrast, flair, post_contrast], axis=-1))
            
            mask_file = file.replace('.tif', '_mask.tif')
            if os.path.exists(os.path.join(patient_folder, mask_file)):
                mask = imread(os.path.join(patient_folder, mask_file))
                masks.append(mask)

    return np.array(images), np.array(masks)

def preprocess_and_augment_data(dataset_dir, img_size=(128, 128)):
    all_images = []
    all_masks = []

    for patient_folder in os.listdir(dataset_dir):
        patient_dir = os.path.join(dataset_dir, patient_folder)
        if os.path.isdir(patient_dir):
            images, masks = load_image_and_mask(patient_dir)
            for img, mask in zip(images, masks):
                img_resized = cv2.resize(img, img_size)
                mask_resized = cv2.resize(mask, img_size)
                all_images.append(img_resized)
                all_masks.append(mask_resized)

    all_images = np.array(all_images)
    all_masks = np.array(all_masks)

    # Normalize the images
    all_images = all_images / 255.0
    all_masks = all_masks / 255.0

    # Split data into training and testing sets (80% training, 20% testing)
    train_images, test_images, train_masks, test_masks = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)

    # Augment training data only
    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2)
    return datagen.flow(train_images, train_masks, batch_size=32), (test_images, test_masks)
