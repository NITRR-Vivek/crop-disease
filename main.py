from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Load the saved model
model = tf.keras.models.load_model('tomato_disease_model2.keras')

# Define image dimensions
img_height, img_width = 150, 150

app = FastAPI()

def preprocess_image(img: io.BytesIO) -> np.ndarray:
    img = image.load_img(img, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(img_array: np.ndarray) -> str:
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # List of class labels
    class_labels = [
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img = io.BytesIO(await file.read())
        img_array = preprocess_image(img)
        prediction = predict_image(img_array)
        return JSONResponse(content={"predicted_class": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
