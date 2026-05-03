from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import numpy as np
import os
import pandas as pd
from skimage import io, img_as_ubyte


def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extracts a Local Binary Pattern (LBP) histogram from an image. Expects a NumPy array image (either rgb or greyscale).
    """
    # LBP requires a 2d greyscale image
    if len(image.shape) == 3:
        image = rgb2gray(image)

    # convert floats to integers for better accuracy and fewer warnings
    image = img_as_ubyte(image)

    # Extract LBP image
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    # Calculate histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normalise histogram so different image sizes don't break the maths (for 64x64 vs 128x128)
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7  # to prevent division by 0

    return hist


def generate_lbp_dataframe(
    metadata_df: pd.DataFrame, base_image_path="./datasets/task1_data"
):
    """
    Loops through the metadata df, loads images, and extracts its LBP features, returning them as a pandas DataFrame.
    """
    lbp_features_list = []
    image_ids = []

    print("Starting LBP feature extraction")
    total_images = len(metadata_df)

    # loop over every image in metadata_df
    for index, row in metadata_df.iterrows():
        img_id = row["image_id"]
        # create exact path to that image
        img_path = os.path.join(base_image_path, row["image_path"])
        try:
            # load image into NumPy array
            image = io.imread(img_path)
            lbp_hist = extract_lbp_features(image)
            lbp_features_list.append(lbp_hist)
            image_ids.append(img_id)
        except Exception as e:
            print(f"\nFailed to load image at {img_path}: {e}")
        if (index + 1) % 100 == 0:
            print(f"Extracted {index + 1}/{total_images} images...", end="\r")
    print("LBP extraction complete!")
    # convert list of lists into pandas DataFrame
    num_lbp_features = len(lbp_features_list[0])
    col_names = [f"lbp_{i}" for i in range(num_lbp_features)]
    lbp_df = pd.DataFrame(lbp_features_list, columns=col_names)

    # add image_id back in so we can merge on it
    lbp_df["image_id"] = image_ids
    return lbp_df
