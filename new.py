from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model,model_from_json

json_file =open('facialemotionmodel.json','r')
model_json =json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('facialemotionmodel.h5')

label = ['angry','disgust','fear','happy','neutral','sad','surprise']

def ef(image):
    img= load_img(image,grayscale=True)
    feature =np.array(img)
    feature=feature.reshape(1,48,48,1)
    return feature/255.0

image = 'data/archive (6)/train/angry/Training_3908.jpg'
print('original image is of angry')
img = ef(image)
pred= model.predict(img)
pred_label =label[pred.argmax()]
print('model prediction is ',pred_label)