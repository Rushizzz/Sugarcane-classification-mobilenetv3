import tensorflow as tf
import cv2
import numpy as np
import time

np.set_printoptions(suppress=True)

model = tf.keras.models.load_model('sugarcane_MobileNetSmall_Custom.h5')

with open('labels.txt', 'r') as f:
    class_names = f.read().split('\n')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224,224)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    start = time.time()

    ret, img = cap.read()

    height, width, channels = img.shape

    scale_value = width/height

    img_resized = cv2.resize(img, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)

    img_array = np.asarray(img_resized)

    normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
    
    data[0] = normalized_img_array

    prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.putText(img, class_name, (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.putText(img, str(float("{:.2f}".format(confidence_score*100))) + "%", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    cv2.imshow('Classification Original', img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()