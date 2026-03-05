import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# load trained model
model = tf.keras.models.load_model("model/fashion_cnn_model.h5")

# class names
class_names = [
"T-shirt","Trouser","Pullover","Dress",
"Coat","Sandal","Shirt","Sneaker",
"Bag","Ankle boot"
]

st.title("Fashion Classifier")

uploaded = st.file_uploader("Upload image")

if uploaded:

    # open image
    img = Image.open(uploaded)

    st.subheader("Original Image")
    st.image(img, width=200)

    # preprocessing
    img = img.convert("L")          # grayscale
    img = img.resize((28,28))       # resize

    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    # reshape for CNN
    img_array = img_array.reshape(1,28,28,1)

    st.subheader("Processed Image (28x28)")
    st.image(img_array.reshape(28,28), width=200)

    # prediction
    pred = model.predict(img_array)[0]

    predicted_class = class_names[np.argmax(pred)]

    confidence = np.max(pred) * 100

    st.subheader("Prediction")
    st.write("Class:", predicted_class)

    st.write("Confidence:", round(confidence,2), "%")

    # Top 3 predictions
    st.subheader("Top 3 Predictions")

    top3 = pred.argsort()[-3:][::-1]

    for i in top3:
        st.write(class_names[i], ":", round(pred[i]*100,2), "%")