import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from torchvision.models import resnet18
from torchvision import transforms
from train.feature_extraction import extract_features_per_sample
import joblib

### Add this as a transofrmation to pre-process the images
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA.pt", map_location="cpu")

model.load_state_dict(ckpt)

# Define normalization transform ONLY
image_transform = transforms.Normalize(mean=mean, std=std)

#### DATASETS
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []    # Should contain PIL.Image or raw image tensors
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


private_data = torch.load("priv_out.pt", weights_only=False)
private_data.transform = image_transform

memberships_classifier = joblib.load('logistic_model.pkl')

features, labels, memberships = extract_features_per_sample(model, private_data)
if not isinstance(features, np.ndarray):
    features = features.numpy()

# Get the predicted probabilities for the positive class (class 1)
scores = memberships_classifier.predict_proba(features)[:, 1]

# Create the submission DataFrame
df = pd.DataFrame(
    {
        "ids": private_data.ids,
        "score": scores,
    }
)

# Save to CSV
df.to_csv("test.csv", index=False)
response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "54614611"})
print(response.json())
