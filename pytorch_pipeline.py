import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from constants import TASK2_DATA


def create_unfrozen_resnet(num_classes=10):
    print("Downloading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # unfreeze all convolutional layers so they can learn
    for param in model.parameters():
        param.requires_grad = True
    # swap the last layer (the 'head') from 1000 classes to 10 classes
    num_features = (
        model.fc.in_features
    )  # fc means fully connected, or the network's output layer
    model.fc = nn.Linear(num_features, num_classes)
    return model


class ImageDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_dir, transform=None):
        """
        metadata_df: pandas DataFrame containing 'image_path' and 'encoded_label' (0-9).
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_path = os.path.join(self.base_dir, self.metadata.loc[index, "image_path"])
        label = self.metadata.loc[index, "encoded_label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label


def train_resnet_model(model, train_loader, val_loader, epochs=10):
    """
    Trains an unfrozen ResNet model with different parameters and checks its results on a validation
    set, to find what the best weights are.
    """
    # move model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # set up the loss function
    criterion = nn.CrossEntropyLoss()

    # set up differential learning rates
    # small changes for the base (cos it's already smart), fast learning for the head
    optimiser = torch.optim.Adam(
        [
            {"params": model.conv1.parameters(), "lr": 1e-5},
            {"params": model.layer1.parameters(), "lr": 1e-5},
            {"params": model.layer2.parameters(), "lr": 1e-5},
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ]
    )

    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"Started training on {device}...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 15)

        # --- training phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # backward pass and optimise
            loss.backward()
            optimiser.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)

        # --- validation phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        # torch.nograd() tells the model not to bother tracking how to improve. it's eval time not learning time
        with torch.no_grad():
            for inputs, labels in val_loader:  # model has never seen these before
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
    print(f"\nTraining complete! Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_weights)
    return model


def run_resnet_training(
    epochs=30, batch_size=32, weights_path="./finetuned_resnet_weights.pth"
):
    """
    Checks for existing weights. If none exist, prepares data and trains the ResNet.
    """

    if os.path.exists(weights_path):
        print(f"Saved weights already exist at '{weights_path}'. Skipping training.")
        print("If you want to retrain, delete or rename the existing .pth file first.")
        return
    print("No saved weights found. Preparing data for training...")

    metadata = pd.read_csv(f"./datasets/{TASK2_DATA}/train_metadata.csv")
    # encode bird labels to integers 0-9
    le = LabelEncoder()
    metadata["encoded_label"] = le.fit_transform(metadata["class_name"])

    train_df, val_df = train_test_split(
        metadata, test_size=0.2, random_state=2718, stratify=metadata["encoded_label"]
    )

    # image preprocessing that ImageNet requires
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # build data loaders
    train_dataset = ImageDataset(
        train_df, base_dir=f"./datasets/{TASK2_DATA}/", transform=preprocess
    )
    val_dataset = ImageDataset(
        val_df, base_dir=f"./datasets/{TASK2_DATA}/", transform=preprocess
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initialising unfrozen ResNet...")
    model = create_unfrozen_resnet(num_classes=10)

    best_model = train_resnet_model(
        model, train_loader=train_loader, val_loader=val_loader, epochs=epochs
    )
    torch.save(best_model.state_dict(), weights_path)
    print(f"Training complete and new weights saved to {weights_path}.")
