from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

app = FastAPI()

model = load_model('tomato_disease_model2.keras')

img_height, img_width = 150, 150

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    img = image.load_img(file.filename, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])

    class_labels = [
        'Bacterial_spot',
        'Early_blight',
        'Late_blight',
        'Leaf_Mold',
        'Septoria_leaf_spot',
        'Spider_mites Two-spotted_spider_mite',
        'Target_Spot',
        'Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato_mosaic_virus',
        'healthy']
    predicted_class = class_labels[predicted_class_index]

    os.remove(file.filename)  # Clean up the uploaded file

    return {"predicted_class": predicted_class}
