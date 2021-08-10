from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from core import datasets
from core import models
import numpy as np
import argparse
import locale
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="../dataset/Houses-dataset/Houses Dataset/", help="path to input dataset of house images")
args = vars(ap.parse_args())

print("[INFO] Loading house sttributes")
input_path = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(input_path)

print("[INFO] Constructing training/testing split")
train, test = train_test_split(df, test_size=0.25, random_state=42)

max_price = train["price"].max()
y_train = train["price"] / max_price
y_test = test["price"] / max_price

print("[INFO] Processing Data...")
x_train, x_test = datasets.process_house_attributes(df, train, test)

model = models.create_mlp(x_train.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] Training Model...")
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=200, batch_size=8)

print("[INFO] Predicting House Prices...")
preds = model.predict(x_test)

diff = preds.flatten() - y_test
percent_diff = (diff - y_test) * 100
abs_percent_diff = np.abs(percent_diff)

mean = np.mean(abs_percent_diff)
std = np.std(abs_percent_diff)

print("[INFO] Avg. house price: {}, STD house price: {}".format(df["price"].mean(), df["price"].std()))
print("[INFO] Mean: {:.2f}%, STD: {:.2f}%".format(mean, std))
