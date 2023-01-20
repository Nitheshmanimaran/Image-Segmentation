MODEL_PATH = 'D:/EPITA/Computer Vision/Project_3/models/model_v1.h5'
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)

model = keras.models.load_model(MODEL_PATH, custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})
def preprocess(path):
    img = Image.open(path)
    # resize the image to 128x128
    img = img.resize((128, 128))
    img = np.array(img) / 255.
    img = np.reshape(img, (1, 128, 128, 3))
    return img

#Author
st.title('Author')
st.write('This project was made by - Nithesh Kumar')

#Getting the image from the user
st.title('Image Segmentation')
st.write('Please upload an image to segment it')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = preprocess(uploaded_file)
    pred = model.predict(img)
    y_pred = np.argmax(pred, axis=-1)
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred, (128, 128))

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))
    img = np.reshape(img, (128, 128, 3))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(y_pred, cmap="viridis")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    st.pyplot(fig)