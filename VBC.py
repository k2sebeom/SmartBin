import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime

def prob(result):
    T = np.argmax(result)
    P = np.amax(result)
    stat = str(P * 100 // 1) + "% "
    if T == 0:
        stat += "CardBoard"
    elif T == 1:
        stat += "Glass"
    elif T == 2:
        stat += "Metal"
    elif T == 3:
        stat += "Paper"
    elif T == 4:
        stat += "Plastic"
    elif T == 5:
        stat += "Trash"
    return stat

#1. Data Load
X_Set = np.load("./DataSet/TrainSet.npy", allow_pickle=True)
Y_Set = np.load("./DataSet/LabelSet.npy", allow_pickle=True)

#2. Split into Train set and Test set
n_images = X_Set.shape[0]
shuffled_i = np.random.permutation(n_images)
split_i = int(n_images * 0.9)
X_Train = X_Set[shuffled_i[:split_i]]
Y_Train = Y_Set[shuffled_i[:split_i]]
X_Test = X_Set[shuffled_i[split_i:]]
Y_Test = Y_Set[shuffled_i[split_i:]]

#3. Setting a model
model = Sequential()
KERNEL = (3,3)
Input_Size = (X_Set.shape[1], X_Set.shape[2], X_Set.shape[3])

model.add(Conv2D(20, KERNEL, input_shape=Input_Size))
model.add(Conv2D(50, KERNEL))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100, KERNEL))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(120, KERNEL, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#4. Hyperparameters and Fit
EPOCHS = 50
BATCH_SIZE = 20

PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0,mode='auto')
LOG_DIR_ROOT="./summaries/"
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}run-{}".format(LOG_DIR_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

callbacks = [early_stopping, tensorboard]

model.fit(X_Train, Y_Train,
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1,
          validation_data=(X_Test, Y_Test))
model.save('./models/run-'+now+'.h5')
#5. Validation
results = model.predict(X_Test)
n = np.random.choice(len(X_Test),1)
RImage = cv2.resize(X_Test[n[0]], (200,200))
cv2.putText(RImage, prob(results[n[0]]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                         0.7, (0,0,0), lineType=cv2.LINE_AA)
cv2.imshow("Result", RImage)
cv2.waitKey(0)


    