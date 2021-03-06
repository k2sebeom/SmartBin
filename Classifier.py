import cv2
import numpy as np
from keras.models import load_model
import time

def prob(result):
    T = np.argmax(result)
    P = np.amax(result)
    stat = str(P * 100 // 1) + "% "
    if T == 0:
        stat += "Glass"
    elif T == 1:
        stat += "Metal"
    elif T == 2:
        stat += "Paper"
    elif T == 3:
        stat += "Plastic"
    elif T == 4:
        stat += "Trash"
    return stat

model = load_model('./models/model.h5')
status = ""
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    Im_Trash = frame
    cv2.putText(Im_Trash, status, (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                         0.7, (0,0,0), lineType=cv2.LINE_AA)
    cv2.imshow("window",Im_Trash)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)
    Im_Trash = cv2.resize(frame, (50,50), interpolation=cv2.INTER_LINEAR) / 255
    result = model.predict(np.array([Im_Trash]))
    status = prob(result[0])
    print(result[0])
    
cap.release()
cv2.destroyAllWindows()