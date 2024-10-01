from preprocessing import preprocess_and_augment_data
from unet_plus_plus import unet_plus_plus
from attention_unet import attention_unet
from sklearn.metrics import f1_score

# Load and preprocess data
data_dir = "data"  # Path to the dataset
train_data_gen, (test_images, test_masks) = preprocess_and_augment_data(data_dir)

# Train Nested U-Net (U-Net++)
print("Training Nested U-Net (U-Net++)...")
unet_plus_model = unet_plus_plus(input_shape=(128, 128, 3))
unet_plus_model.fit(train_data_gen, epochs=4, validation_data=(test_images, test_masks))
unet_plus_model.save('models/unet_plus_plus.keras')  # Save in the new Keras format

# Train Attention U-Net
print("Training Attention U-Net...")
attention_unet_model = attention_unet(input_shape=(128, 128, 3))
attention_unet_model.fit(train_data_gen, epochs=4, validation_data=(test_images, test_masks))
attention_unet_model.save('models/attention_unet.keras')  # Save in the new Keras format

# Evaluate the models using DICE score (based on F1 score for segmentation)
def dice_score(y_true, y_pred):
    return f1_score(y_true.flatten(), y_pred.flatten())

# Evaluate Nested U-Net
y_pred_unet = unet_plus_model.predict(test_images)
dice_unet = dice_score(test_masks, y_pred_unet)
print(f"Dice Score (Nested U-Net): {dice_unet}")

# Evaluate Attention U-Net
y_pred_att_unet = attention_unet_model.predict(test_images)
dice_att_unet = dice_score(test_masks, y_pred_att_unet)
print(f"Dice Score (Attention U-Net): {dice_att_unet}")
