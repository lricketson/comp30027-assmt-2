import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from constants import LEAKAGE_COLS
import numpy as np


def create_hog_colour_df():
    train_metadata = pd.read_csv("./datasets/task1_data/train_metadata.csv")
    hog_pca = pd.read_csv("./datasets/task1_data/hog_pca.csv")
    colour_df = pd.read_csv("./datasets/task1_data/color_histogram.csv")
    temp_df = pd.merge(left=hog_pca, right=train_metadata, on="image_id", how="inner")
    colour_hog_df = pd.merge(left=colour_df, right=temp_df, on="image_id", how="inner")
    return colour_hog_df


def scale_data(X_train, X_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_test)

    # StandardScaler also converts these pd DataFrames into NumPy arrays, which we want for the NN.

    return X_train_scaled, X_val_scaled


def one_hot_encode(Y_ints: np.ndarray, num_classes: int):
    m = Y_ints.shape[0]
    Y_ohe = np.zeros((m, num_classes))
    Y_ohe[np.arange(m), Y_ints] = 1

    return Y_ohe
