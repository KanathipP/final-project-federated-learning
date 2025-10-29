import sys
from pathlib import Path
import numpy as np
import mne
import torch
import importlib
from sklearn.model_selection import StratifiedShuffleSplit  # <-- เปลี่ยนตรงนี้
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

mne.set_log_level("ERROR")

BASE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = Path(BASE_DIR / "../data")

def read_data(path):
    raw = mne.io.read_raw_gdf(path, 
                              preload=True,
                              eog=['EOG-left', 'EOG-central', 'EOG-right'],
                              verbose='ERROR',
                             )
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.set_eeg_reference(verbose="ERROR")
    events = mne.events_from_annotations(raw)
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=[5, 6, 7, 8],
        on_missing='warn',
        verbose="ERROR",
    )

    features = epochs.get_data()
    labels = epochs.events[:,-1]
    return features,labels

#https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
class StandardScaler3D(BaseEstimator,TransformerMixin):
    #batch, sequence, channels
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self,X,y=None):
        self.scaler.fit(X.reshape(-1, X.shape[2]))
        return self

    def transform(self,X):
        return self.scaler.transform(X.reshape( -1,X.shape[2])).reshape(X.shape)

def remap_np(y: torch.Tensor) -> torch.Tensor:
    y_np = y.view(-1).cpu().numpy()
    uniq = np.unique(y_np)                      
    lut = {u: i for i, u in enumerate(uniq)}     
    y_new = np.vectorize(lut.get)(y_np)
    return torch.as_tensor(y_new, dtype=torch.long, device=y.device)


def pipeline(partition_id: int):
    features, labels = read_data(Path(EXTRACT_DIR/ f'{partition_id}/A0{partition_id}T.gdf'))
    features = np.moveaxis(features, 1, 2)  

    sss_test = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
    train_val_index, test_index = next(sss_test.split(features, labels))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    relative_train_index, relative_val_index = next(sss_val.split(features[train_val_index], labels[train_val_index]))
    train_index = train_val_index[relative_train_index]
    val_index = train_val_index[relative_val_index]

    train_features, train_labels = features[train_index], labels[train_index]
    val_features,  val_labels  = features[val_index],  labels[val_index]
    test_features,  test_labels  = features[test_index],  labels[test_index]

    scaler = StandardScaler3D()
    train_features = scaler.fit_transform(train_features)
    val_features  = scaler.transform(val_features)
    test_features  = scaler.transform(test_features)

    train_features = np.moveaxis(train_features, 1, 2)
    val_features = np.moveaxis(val_features, 1, 2)
    test_features  = np.moveaxis(test_features, 1, 2)

    train_features = torch.Tensor(train_features)
    val_features = torch.Tensor(val_features)
    test_features = torch.Tensor(test_features)

    train_labels = torch.Tensor(train_labels)
    val_labels = torch.Tensor(val_labels)
    test_labels = torch.Tensor(test_labels)

    train_labels = remap_np(train_labels)
    val_labels = remap_np(val_labels)
    test_labels   = remap_np(test_labels)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


