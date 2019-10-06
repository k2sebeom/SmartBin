import serial
import cv2
import numpy as np
from keras.models import load_model
import time
import glob

model = load_model('./models/run-20190922070535.h5')
ser = serial.Serial("COM4", 9600)
time.sleep(3)
ser.write("40".encode())
input("start?")
for fpath in np.random.permutation(glob.glob("./Test/*.jpg")):
    imo = cv2.imread(fpath)
    im = cv2.resize(imo, (50,50), interpolation=cv2.INTER_LINEAR) / 255
    result = model.predict(np.array([im]))
    i = np.argmax(result[0])
    if(i == 1):
        ser.write("10".encode())
    elif(i == 2):
        ser.write("70".encode())
    elif(i == 4):
        ser.write("40".encode())
    cv2.imshow("result", imo)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
