import cv2
import numpy as np
import glob
import os.path as path

IM_DIR = glob.glob("./Raw_Images/*/*")
images = []
labels = []
for im in IM_DIR:
    image = cv2.imread(im)
    image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    images.append(image)
    cv2.imwrite("./results/"+path.basename(im)+".png", image)
    name = path.basename(im)[0:2]
    if name == "ca":
        labels.append(np.array([1,0,0,0,0,0]))
    elif name == "gl":
        labels.append(np.array([0,1,0,0,0,0]))
    elif name == "me":
        labels.append(np.array([0,0,1,0,0,0]))
    elif name == "pa":
        labels.append(np.array([0,0,0,1,0,0]))
    elif name == "pl":
        labels.append(np.array([0,0,0,0,1,0]))
    elif name == "tr":
        labels.append(np.array([0,0,0,0,0,1]))
    print("{}%".format(int(len(images)/len(IM_DIR)*100)), end="\r")
    
images = np.array(images) / 255
labels = np.array(labels)
    
np.save("./DataSet/TrainSet", images, allow_pickle=True)
np.save("./DataSet/LabelSet", labels, allow_pickle=True)

    
