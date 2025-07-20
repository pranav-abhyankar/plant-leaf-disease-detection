from flask import Flask, request, url_for, render_template, redirect
import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import re

app = Flask(__name__, template_folder='template', static_folder='static')

# Defining class names
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
              'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load your trained machine learning model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Loading the necessary DataFrames
df_desc = pd.read_csv('data_description')
df_disease = pd.read_csv('data_disease')

# Creating a mapping
precedence = {'Apple': 0, 'Blueberry': 1, 'Cherry': 2, 'Corn': 3, 'Grape': 4, 'Orange': 5, 'Peach': 6, 'Pepper,': 7,
              'Potato': 8, 'Raspberry': 9, 'Soybean': 10, 'Squash': 11, 'Strawberry': 12, 'Tomato': 13}

# Create folder to store files if doesn't already exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
	os.makedirs(UPLOAD_FOLDER)


# Process the uploaded image
def predict_disease(image_path):
	img = cv2.imread(image_path)
	cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
	image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
	input_arr = tf.keras.preprocessing.image.img_to_array(image)
	input_arr = np.array([input_arr])  # Convert single image to a batch.
	predictions = model.predict(input_arr)
	result_index = np.argmax(predictions)
	return class_name[result_index]


@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		if 'image' not in request.files:
			return render_template('index.html', name='No file part')

		file = request.files['image']

		if file.filename == '':
			return render_template('index.html', name='No selected file')

		if file:
			filename = os.path.join(UPLOAD_FOLDER, file.filename)
			file.save(filename)
			prediction = predict_disease(filename)
			return redirect(url_for('result', cls=prediction, filename=file.filename))
	return render_template('index.html')


@app.route('/results/<string:cls>/<string:filename>')
def result(cls, filename):
	plant_name = ""
	i = 0
	while cls[i] != '_':
		plant_name += cls[i]
		i += 1

	desc = df_desc.iloc[precedence[plant_name], 1]

	if cls[-7:] == 'healthy':
		return render_template('results.html', cls=cls, filename=filename, plant_name=plant_name, desc=desc)

	info, why, remedies = df_disease[cls].values
	remedies = re.findall(r"'(.*?)'", remedies)
	return render_template('results.html', cls=cls, filename=filename, plant_name=plant_name, desc=desc, info=info, why=why, remedies=remedies)


if __name__ == '__main__':
	app.run(debug=True)