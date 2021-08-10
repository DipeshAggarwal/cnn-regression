from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_house_attributes(input_path, min_count=25):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(input_path, sep=" ", header=None, names=cols)
    
    # Find all unique zipcode
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    # Count how many times unique zipcode occurs
    counts = df["zipcode"].value_counts().tolist()
    
    for zipcode, count in zip(zipcodes, counts):
        # Find the index of all the instances of a zipcode which has less
        # than `min_count` entries and remove them
        if count < min_count:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)
            
    return df

def process_house_attributes(df, train, test):
    continous = ["bedrooms", "bathrooms", "area"]
    
    sc = MinMaxScaler()
    train_continous = sc.fit_transform(train[continous])
    test_continous = sc.transform(test[continous])
    
    zip_lb = LabelBinarizer().fit(df["zipcode"])
    train_categorical = zip_lb.transform(train["zipcode"])
    test_categorical = zip_lb.transform(test["zipcode"])
    
    x_train = np.hstack([train_categorical, train_continous])
    x_test = np.hstack([test_categorical, test_continous])
    
    return (x_train, x_test)
