from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from keras.models import load_model
from tifffile import imread

app = FastAPI()

model = load_model("models/unet_plus_plus.h5")  # Load the best model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    image = imread(image)
    
    # Preprocess the image
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    
    # Predict
    pred_mask = model.predict(image_resized)
    
    return {"prediction": pred_mask.tolist()}
