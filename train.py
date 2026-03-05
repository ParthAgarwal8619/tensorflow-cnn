import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# load dataset
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

# normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for CNN
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)

# class names
class_names = [
    "T-shirt","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker",
    "Bag","Ankle boot"
]

# CNN MODEL
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dense(10,activation="softmax")
])

# compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

# TRAIN MODEL
history = model.fit(train_images, train_labels, epochs=10)

# EVALUATE MODEL
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc)

# SAVE MODEL
model.save("model/fashion_cnn_model.h5")

print("Model saved successfully")

# ----------------------------
# TRAINING GRAPH
# ----------------------------

plt.figure()

plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['loss'], label="Training Loss")

plt.title("Training Graph")
plt.xlabel("Epoch")
plt.ylabel("Value")

plt.legend()

plt.show()

# ----------------------------
# CONFUSION MATRIX
# ----------------------------

pred = model.predict(test_images)
pred_labels = pred.argmax(axis=1)

cm = confusion_matrix(test_labels, pred_labels)

plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()