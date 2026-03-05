import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# class names
class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]

# load dataset
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

# normalize
test_images = test_images / 255.0

# reshape
test_images = test_images.reshape(-1,28,28,1)

# load model
model = tf.keras.models.load_model("model/fashion_cnn_model.h5")

# random image
index = random.randint(0,10000)
img = test_images[index]

# prediction
prediction = model.predict(img.reshape(1,28,28,1))[0]

# top 3 predictions
top3 = prediction.argsort()[-3:][::-1]

print("\nTop Predictions:\n")

for i in top3:
    print(class_names[i], ":", round(prediction[i]*100,2),"%")

# show image
plt.imshow(img.reshape(28,28), cmap="gray")
plt.title("Predicted: " + class_names[top3[0]])
plt.show()