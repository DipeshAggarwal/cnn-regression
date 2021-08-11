from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from core import datasets
from core import models
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="../dataset/Houses-dataset/Houses Dataset/", help="path to input dataset of house images")
args = vars(ap.parse_args())

print("[INFO] Loading House Attributes...")
input_path = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(input_path)

print("[INFO] Loading House Images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0

split = train_test_split(df, images, test_size=0.25, random_state=42)
x_train_attr, x_test_attr, x_train_images, x_test_images = split

max_price = x_train_attr["price"].max()
y_train = x_train_attr["price"] / max_price
y_test = x_test_attr["price"] / max_price

model = models.create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] Trining Model...")
model.fit(x=x_train_images, y=y_train, validation_data=(x_test_images, y_test), epochs=200, batch_size=8)

print("[INFO] Predicting House Prices...")
preds = model.predict(x_test_images)

diff = preds.flatten() - y_test
percent_diff = (diff - y_test) * 100
abs_percent_diff = np.abs(percent_diff)

mean = np.mean(abs_percent_diff)
std = np.std(abs_percent_diff)

print("[INFO] Avg. house price: {}, STD house price: {}".format(df["price"].mean(), df["price"].std()))
print("[INFO] Mean: {:.2f}%, STD: {:.2f}%".format(mean, std))
