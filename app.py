from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os


from googletrans import Translator
from gtts import gTTS
from flask import send_file

translator = Translator()



app = Flask(__name__)

model = tf.keras.models.load_model("models/plant_disease_prediction_model.keras")

# Load disease JSON
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')



#voice translation function

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get("text")
    lang = data.get("lang")  

    # Translate text
    translated = translator.translate(text, dest=lang).text

    # Generate voice file
    tts = gTTS(text=translated, lang=lang)
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("uploadimages", filename)
    tts.save(filepath)

    return {"audio_url": f"/uploadimages/{filename}"}



def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    index = prediction.argmax()

    # Extract name from JSON
    data = plant_disease[index]
    name = data["name"]
    cause = data["cause"] 
    cure = data["cure"]

    return name, cause, cure


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']

        filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        save_path = os.path.join("uploadimages", filename)
        image.save(save_path)

        # Get prediction details
        name,cause,cure = model_predict(save_path)

        return render_template(
            'home.html',
            result=True,
            imagepath=f"/uploadimages/{filename}",
            name=name,
            cause=cause,
            cure=cure
        )

    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
