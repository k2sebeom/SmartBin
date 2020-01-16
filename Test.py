import cv2
import numpy as np
from keras.models import load_model
import time
import sys

def prob(result):
    T = np.argmax(result)
    P = np.amax(result)
    stat = str(P * 100 // 1) + "% "
    if T == 0:
        stat += "Glass"
    elif T == 1:
        stat += "Metal"
    elif T == 2:
        stat += "Plastic"
    elif T == 3:
        stat += "Trash"
    elif T == 4:
        stat += "Paper"
    return stat

model = load_model('./models/FullBatch.h5')
status = ""
path = input("File path: ")
while path != "quit":
    frame = cv2.imread(path)
    Im_Trash = cv2.resize(frame, (100,100), interpolation=cv2.INTER_LINEAR)/255
    result = model.predict(np.array([Im_Trash]))
    status = prob(result[0])
    cv2.putText(frame, status, (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                         0.7, (0,0,0), lineType=cv2.LINE_AA)
    print(result[0])
    cv2.imshow("window",frame)
    cv2.waitKey(0)
    path = input("File path: ")

cv2.destroyAllWindows()