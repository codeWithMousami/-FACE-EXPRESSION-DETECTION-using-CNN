import cv2
import os
from keras.models import model_from_json
import numpy as np

# Load the Keras model
json_file = open('facialemotionmodel.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('facialemotionmodel.h5')

# Get the OpenCV data directory
cv2_data_dir = cv2.data.haarcascades
print("OpenCV data directory:", cv2_data_dir)

# Specify the full path to the haarcascade_frontalface_default.xml file
haar_file = os.path.join(cv2_data_dir, 'haarcascade_frontalface_default.xml')
print("Haar cascade file:", haar_file)

# Create the face cascade classifier
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Attempt to open the webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Failed to open webcam")
else:
    print("Webcam opened successfully")

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read frame from the webcam
    i, im = webcam.read()
    if not i:
        print("Error: Failed to read frame from webcam")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            # Extract face region
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow('Output', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error:
        print("Error: OpenCV error occurred")
        pass

# Release webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
