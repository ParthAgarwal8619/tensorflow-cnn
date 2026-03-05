# Fashion MNIST CNN Classifier

A Deep Learning based **Fashion Image Classifier** built using **TensorFlow CNN** and deployed as a **web app using Streamlit**.

This project classifies clothing images into categories such as **T-shirt, Shirt, Coat, Sneaker, Bag, etc.**

---

## 🚀 Live Demo

You can try the live app here:

https://flask2project-pc3hazmlt5ba6fz4zts9mh.streamlit.app/

Upload an image and the AI model will predict the clothing category.

---

## 🧠 Model Information

Dataset used: **Fashion MNIST**

Number of classes: **10**

Classes:

* T-shirt
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

Model type: **Convolutional Neural Network (CNN)**

Accuracy: ~93-96%

---

## ⚙️ Features

* Image upload prediction
* CNN deep learning model
* Image preprocessing (resize + grayscale)
* Confidence score display
* Top-3 predictions
* Processed image preview
* Web interface built with Streamlit

---

## 🏗️ Project Workflow

Dataset → Preprocessing → CNN Model → Training → Model Saving → Streamlit Web App → Deployment

---

## 📂 Project Structure

```
fashion-classifier
│
├── model
│   └── fashion_cnn_model.h5
│
├── demo
│   ├── demo1.png
│   └── demo2.png
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## 🖥️ Installation

Clone the repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

Go to project directory

```
cd fashion-classifier
```

Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Run the Streamlit app

```
streamlit run app.py
```

The application will start at

```
http://localhost:8501
```

---

## 🧪 Training the Model

To train the CNN model run:

```
python train.py
```

This will train the neural network and save the model as:

```
model/fashion_cnn_model.h5
```

---

## 📸 Demo

Example prediction output

(Add screenshots inside the demo folder)

```
demo/demo1.png
demo/demo2.png
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Streamlit

---

## 📊 Model Architecture

CNN Architecture:

```
Conv2D
MaxPooling
Conv2D
MaxPooling
Flatten
Dense
Dense (Softmax)
```

---

## 📌 Future Improvements

* Better CNN architecture
* Support for real clothing images
* Mobile friendly UI
* Deploy with Docker
* Add more datasets

---

## 👨‍💻 Author

Developed as a Deep Learning project using TensorFlow and Streamlit.

---
