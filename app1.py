
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image,ImageOps
from io import BytesIO
import cv2

#@st.cache(allow_output_mutation=True)
st.title('Garbage classifcation')
st.text('This web app classifies  the given image as a cardboard,glass,metal,paper,plastic or a trash')
def load_model():
  model=tf.keras.models.load_model('C:/Users/SHIVA/Desktop/final_garbage_classification_2.hdf5')
  return model
model = load_model()
file = st.file_uploader("please upload a image only jpg and png are suported", type=["jpg","png"])
def import_and_predict(image_data, model):
  size=(300,300)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  prediction = model.predict(img_reshape)

  return prediction
if file is None:
  st.text("please upload a image file")
else:
  image = Image.open(file)
  st.image(image, use_colum_width=True)
  predictions=import_and_predict(image,model)
  class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
  string="This image is most likely is: "+str(class_names[np.argmax(predictions)])
  st.success(string)