from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import numpy as np
import os
import pandas as pd
from skimage import io, img_as_ubyte

# --------------------------------------------- GENERAL UTILITY ---------------------------------------------


def build_feature_dataframe(
    metadata_df, extractor_func, col_prefix, base_dir="./datasets/task1_data/"
):
    """
    A universal engine that loops through images, runs any provided feature extraction
    function, and returns a formatted DataFrame.
    """
    features_list = []
    image_ids = []
    total_images = len(metadata_df)

    print(f"Starting {col_prefix.upper()} feature extraction...")

    for index, row in metadata_df.iterrows():
        img_id = row["image_id"]
        img_path = os.path.join(base_dir, row["image_path"])

        try:
            # The master loop doesn't care what extractor we're using
            # It just hands the path to the function and expects a list of numbers back.
            features = extractor_func(img_path)

            features_list.append(features)
            image_ids.append(img_id)

        except Exception as e:
            print(f"\nFailed on {img_path}: {e}")

        if (index + 1) % 100 == 0:
            print(f"Extracted {index + 1}/{total_images} images...", end="\r")

    print(f"\n{col_prefix.upper()} Extraction Complete!")

    # Dynamically name columns (e.g., lbp_0 or res_0)
    num_features = len(features_list[0])
    col_names = [f"{col_prefix}_{i}" for i in range(num_features)]

    df = pd.DataFrame(features_list, columns=col_names)
    df["image_id"] = image_ids

    return df


# --------------------------------------------- LBP SECTION ---------------------------------------------


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


# --- LBP WRAPPER ---
def run_lbp(img_path):
    """Opens the image via scikit-image and runs LBP."""
    image = io.imread(img_path)
    return extract_lbp_features(image)


# --------------------------------------------- RESNET SECTION ---------------------------------------------

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def get_resnet_extractor():
    """
    Downloads a pre-trained ResNet, chops off the final layer,
    and sets up the exact preprocessing pipeline required by ImageNet.
    """
    print("Loading pre-trained ResNet...")
    # 1. Load the pre-trained weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # 2. Chop off the head!
    # 'model.fc' is the fully connected final layer. We replace it with an Identity
    # layer, which just passes the raw 512-number feature array straight through.
    model.fc = torch.nn.Identity()

    # 3. Lock the model (CRITICAL)
    # We are just extracting, not training. This turns off random dropouts.
    model.eval()

    # 4. The ImageNet Preprocessing Pipeline
    # ResNet mathematically requires images to be exactly 224x224 pixels and
    # color-normalized to these exact mathematical constants.
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, preprocess


def extract_single_image_features(img_path, model, preprocess):
    """Opens a single image and pushes it through the decapitated ResNet."""
    # Torchvision strongly prefers PIL Images over skimage/cv2
    img = Image.open(img_path).convert("RGB")

    # Run the image through the crop/resize/normalize pipeline
    input_tensor = preprocess(img)

    # PyTorch expects a "batch" of images. We only have 1, so we fake a batch
    # dimension, turning shape [3, 224, 224] into [1, 3, 224, 224]
    input_batch = input_tensor.unsqueeze(0)

    # Push it through the network!
    # torch.no_grad() tells PyTorch not to waste RAM tracking calculus gradients.
    with torch.no_grad():
        features = model(input_batch)

    # Return the raw 512 numbers as a flat 1D numpy array
    return features.squeeze().numpy()


# --- RESNET WRAPPER (The Factory Pattern) ---
def create_resnet_extractor():
    """
    Boot up the heavy model ONCE, then return a fast function
    that uses that already-loaded model for the master loop.
    """
    model, preprocess = get_resnet_extractor()

    def run_resnet(img_path):
        return extract_single_image_features(img_path, model, preprocess)

    # Return the function itself, not the result of the function!
    return run_resnet
