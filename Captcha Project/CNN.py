import numpy as np
# from cv2 import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras import models,layers
from sklearn.model_selection import train_test_split
import cv2
import glob
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def load_data_preprocess(dataset):
    all_images = []
    all_labels = []
    for i, item in enumerate(glob.glob(dataset)):
        img = cv2.imread(item)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        all_images.append(img)
        label = item.split("\\")[-2]
        all_labels.append(label)
        if i % 100 == 0:
            print("[INFO] {}/2250 processed".format(i))

    all_images = np.array(all_images)
    le = LabelBinarizer()  # one_hot
    all_labels = le.fit_transform(all_labels)  # transform one_hot
    trainX,testX,trainy,testy=train_test_split(all_images,all_labels,test_size=0.2)
    return trainX,testX,trainy,testy


def miniCNN():
    net=models.Sequential([
        layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation="relu",padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32,activation="relu"),
        layers.Dense(9,activation="softmax")
    ])
    net.compile(loss="categorical_crossentropy",
                optimizer="SGD",
                metrics=["accuracy"])
    return net

def show_learning_curve(H):
    plt.plot(H.history["accuracy"], label="train accuracy")
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["loss"], label="train loss")
    plt.plot(H.history["val_loss"], label="test loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss/accuracy")
    plt.title("Dig classification")
    plt.show()



trainX,testX,trainy,testy=load_data_preprocess("dataset\\*\\*")
net=miniCNN()
H=net.fit(x=trainX,y=trainy,epochs=20,batch_size=32,validation_data=(testX,testy))
show_learning_curve(H)
net.save("digi_classifier.h5")