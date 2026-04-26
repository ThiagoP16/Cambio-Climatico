from flask import Flask, render_template, request
import tf_keras as keras 
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Cargar el modelo 
model = keras.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1. Obtener la imagen del formulario
        file = request.files["archivo_imagen"]
        image = Image.open(file).convert("RGB")
        
        # 2. Preprocesar la imagen (esto es estándar de Teachable Machine)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # 3. Predicción
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return render_template("index.html", resultado=class_name[2:])

    return render_template("index.html")
