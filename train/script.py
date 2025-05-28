import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import pandas as pd
from torchvision.models import resnet18
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import numpy as np
from collections import Counter
from feature_extraction import extract_features_per_sample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib


### Add this as a transofrmation to pre-process the images
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("01_MIA.pt", map_location="cpu")

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

# Example: loading public data and applying transform
public_data = torch.load("pub.pt", weights_only=False)
public_data.transform = image_transform

labels = np.array(public_data.labels)
membership = np.array(public_data.membership)
tags = np.array([f"{l}_{m}" for l, m in zip(labels, membership)])
all_indices = np.arange(len(public_data))

# Count tag frequencies
tag_counts = Counter(tags)

# Separate rare samples (tags with < 2 occurrences)
rare_indices = np.array([i for i, tag in enumerate(tags) if tag_counts[tag] < 2])
strat_eligible = np.array([i for i, tag in enumerate(tags) if tag_counts[tag] >= 2])
strat_tags = tags[strat_eligible]

# Stratified train-test split (e.g., 80% train, 20% test)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx_r, test_idx_r = next(sss.split(strat_eligible, strat_tags))

# Map back to full dataset indices
train_idx_r = strat_eligible[train_idx_r]
test_idx_r = strat_eligible[test_idx_r]

# Combine rare indices with train set (rare samples only go to train to keep them in dataset)
train_idx = np.concatenate([rare_indices, train_idx_r])
test_idx = test_idx_r

# Wrap in Subset for PyTorch DataLoader compatibility
train_data = Subset(public_data, train_idx)
test_data = Subset(public_data, test_idx)

print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

features, labels, memberships = extract_features_per_sample(model, public_data)

# Prepare numpy arrays (assuming features and labels are numpy arrays)
X_train = features[train_idx]
y_train = memberships[train_idx]

X_test = features[test_idx]
y_test = memberships[test_idx]

# Assuming X_train, y_train, X_test, y_test are already defined and scaled as before
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.001, 0.005, 0.007, 0.009, 0.01, 0.1,],          # Regularization strength
    'l1_ratio': [0, 0.005, 0.001, 0.1, 0.5, 0.7],  # Elastic net mixing parameter (0 = L2, 1 = L1)
    'max_iter': [250, 300, 350, 450, 500, 1000],          # Maximum iterations to converge
    # 'penalty' must be elasticnet when using l1_ratio with saga solver, so no need to grid search penalty here
    # Solver must be saga for elasticnet penalty, so keep fixed
}

clf = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    class_weight='balanced',
    multi_class='auto',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

best_clf = grid_search.best_estimator_

# Evaluate on test data
y_pred = best_clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

joblib.dump(best_clf, 'logistic_model.pkl')
