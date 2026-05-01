import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_hog_colour_df():
    train_metadata = pd.read_csv("./datasets/task1_data/train_metadata.csv")
    hog_pca = pd.read_csv("./datasets/task1_data/hog_pca.csv")
    colour_df = pd.read_csv("./datasets/task1_data/color_histogram.csv")
    temp_df = pd.merge(left=hog_pca, right=train_metadata, on="image_id", how="inner")
    colour_hog_df = pd.merge(left=colour_df, right=temp_df, on="image_id", how="inner")
    return colour_hog_df


def scale_data(training_set: pd.DataFrame):
    train_df, val_df = train_test_split(training_set, random_state=2718, test_size=0.2)

    X_train = train_df.drop(
        columns=["image_id", "image_path", "class_name", "class_id"]
    )
    y_train = train_df["class_name"]
    X_val = val_df.drop(columns=["image_id", "image_path", "class_name", "class_id"])
    y_val = val_df["class_name"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # StandardScaler also converts these pd DataFrames into NumPy arrays, which we want for the NN.

    return X_train_scaled, X_val_scaled, y_train, y_val
