import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from constants import LEAKAGE_COLS
import numpy as np
from feature_engineering import (
    build_feature_dataframe,
    run_lbp,
    create_resnet_extractor,
)


def create_tier1_ft_df():
    """
    Creates the tier 1 feature dataset. This is data merged from hog_pca.csv, color_histogram.csv, and
    additional_features.csv.
    """

    train_metadata = pd.read_csv("./datasets/task1_data/train_metadata.csv")
    hog_pca = pd.read_csv("./datasets/task1_data/hog_pca.csv")
    colour_df = pd.read_csv("./datasets/task1_data/color_histogram.csv")
    additional_ft_df = pd.read_csv("./datasets/task1_data/additional_features.csv")

    temp_df = pd.merge(left=hog_pca, right=train_metadata, on="image_id", how="inner")
    temp2_df = pd.merge(left=colour_df, right=temp_df, on="image_id", how="inner")

    tier1_ft_df = pd.merge(
        left=additional_ft_df, right=temp2_df, on="image_id", how="inner"
    )
    return tier1_ft_df


def create_tier2_ft_df():
    """
    Creates the tier 2 feature dataset. This is everything present in tier 1, plus Local Binary Patterns data.
    """
    tier1_ft_df = create_tier1_ft_df()
    metadata_df = pd.read_csv("./datasets/task1_data/train_metadata.csv")

    lbp_df = build_feature_dataframe(
        metadata_df=metadata_df, extractor_func=run_lbp, col_prefix="lbp"
    )
    tier2_ft_df = pd.merge(left=tier1_ft_df, right=lbp_df, on="image_id", how="inner")
    return tier2_ft_df


def create_tier3_ft_df():
    """
    Creates the tier 3 feature dataset. This is features extracted by the pre-trained model ResNet.
    """
    metadata_df = pd.read_csv("./datasets/task1_data/train_metadata.csv")

    resnet_extractor_func = create_resnet_extractor()

    tier3_ft_df = build_feature_dataframe(
        metadata_df=metadata_df, extractor_func=resnet_extractor_func, col_prefix="res"
    )
    tier3_ft_df = pd.merge(
        left=tier3_ft_df, right=metadata_df, on="image_id", how="inner"
    )

    return tier3_ft_df


def scale_data(X_train, X_test, ignore_keywords=None, return_numpy=False):
    """
    Scales training and testing DataFrames, with the ability to selectively ignore columns based
    on keywords (e.g., ignoring columns with the word 'hog' in them).
    """

    if ignore_keywords is None:
        ignore_keywords = []

    cols_to_ignore = [
        col
        for col in X_train.columns
        if any(keyword in col for keyword in ignore_keywords)
    ]
    cols_to_scale = X_train.drop(columns=cols_to_ignore).columns

    scaler = StandardScaler()
    X_train_scaled_part = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled_part = scaler.transform(X_test[cols_to_scale])

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled_part, columns=cols_to_scale, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled_part, columns=cols_to_scale, index=X_test.index
    )

    if cols_to_ignore:
        X_train_final = pd.concat([X_train_scaled_df, X_train[cols_to_ignore]], axis=1)
        X_test_final = pd.concat([X_test_scaled_df, X_test[cols_to_ignore]], axis=1)
    else:
        X_train_final = X_train_scaled_df
        X_test_final = X_test_scaled_df

    if return_numpy:
        return X_train_final.to_numpy(), X_test_final.to_numpy()

    return X_train_final, X_test_final


def one_hot_encode(Y_ints: np.ndarray, num_classes: int):
    m = Y_ints.shape[0]
    Y_ohe = np.zeros((m, num_classes))
    Y_ohe[np.arange(m), Y_ints] = 1

    return Y_ohe
