import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# model= models.load_model('image_classifier.h5')
model= models.load_model('image_classifier_model.keras')


img = cv.imread('car.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap =plt.cm.binary)

prediction = model.predict(np.array([img])/255 )
index = np.argmax(prediction)

print(f"Prediction is {class_names[index]}")