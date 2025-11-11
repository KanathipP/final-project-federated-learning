import sys
from pathlib import Path
import numpy as np
import mne
import torch
import importlib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

mne.set_log_level("ERROR")

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(BASE_DIR / "../../0-raw-data/motor-imaginary")
EXTRACT_DIR = Path(DATASET_DIR / "data")

def download_and_extract_motor_imaginery_data():
    target_dir = DATASET_DIR.resolve()
    if str(target_dir) not in sys.path:
        sys.path.append(str(target_dir))

    import data_fetcher
    importlib.reload(data_fetcher)

    data_fetcher.download_and_extract_data(delete_zip=False)

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
    uniq = np.unique(y_np)                       # e.g., [5,6,7,8]
    lut = {u: i for i, u in enumerate(uniq)}     # {5:0, 6:1, 7:2, 8:3}
    y_new = np.vectorize(lut.get)(y_np)
    return torch.as_tensor(y_new, dtype=torch.long, device=y.device)


def pipeline():
    download_and_extract_motor_imaginery_data()

    features,labels,groups=[],[],[]
    for i in range(1,10):
        feature,label=read_data(Path(EXTRACT_DIR/ f'A0{i}T.gdf'))
        features.append(feature)
        labels.append(label)
        groups.append([i]*len(label))
    
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    groups = np.concatenate(groups)
    features = np.moveaxis(features, 1, 2)
    
    gss_test = GroupShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
    train_val_index, test_index = next(gss_test.split(features, labels, groups))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    relative_train_index, relative_val_index = next(gss_val.split(features[train_val_index], labels[train_val_index], groups[train_val_index]))
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
