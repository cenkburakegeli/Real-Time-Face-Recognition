import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition
prototxt_path = "./weights/deploy.prototxt.txt"

model_path = "./weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
MODEL = "cnn"
# load Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
       
while True:

    # read the desired image
    _, image = cap.read()
    locations = face_recognition.face_locations(image, model = model)
    encodings = face_recognition.face_encodings(image, locations)
    default_color = (255,0,0)
    for face_encoding, face_location in zip(encodings,locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        print(results)
        match = None
        datetime = datetime.now()
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}, {datetime}")
        
    # get width and height of the image
    h, w = image.shape[:2]

        # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(model.forward())
    for i in range(0, output.shape[0]):
        # get the confidence
        confidence = output[i, 2]
        # if confidence is above 45%, then draw the surrounding box
        if confidence > 0.45:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # draw the rectangle surrounding the face  
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), default_color, thickness=2)
            # draw text as well
            cv2.putText(image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, default_color, 2)
            cv2.putText(image, match, (start_x, start_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, default_color, 2)
    # show the image
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()