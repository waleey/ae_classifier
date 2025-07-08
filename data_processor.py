import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

class DataProcessing:
    def __init__(self, file_name, metadata_file=None, **kwargs):
        self.file_name = file_name
        self.metadata_file = metadata_file
        self.input_shape = kwargs.get('input_dataset_shape', 1024)
        self.read_csv()
        self.encode_labels()
        self.split_data_by_tic()
        self.to_tensors()

    def read_csv(self):
        """Reads the main and metadata CSV files."""
        self.data_frame = pd.read_csv(self.file_name)

        if self.data_frame.shape[1] == self.input_shape + 3:
            warnings.warn('Dataset has unwanted extra columns. Please remove them before pre-processing', UserWarning)
            self.data_frame = self.data_frame.drop(columns='Unnamed: 0')

        print(f"Shape of loaded data: {self.data_frame.shape}")

        # TICs and labels
        self.tic_ids = self.data_frame['TIC'].tolist()
        self.labels = self.data_frame['label'].tolist()

        # Strip label and TIC
        self.data_frame = self.data_frame.drop(columns=['TIC', 'label'])
        self.data_orig = self.data_frame.copy()

        # Load metadata if provided
        if self.metadata_file:
            self.meta_df = pd.read_csv(self.metadata_file)

            # Sort metadata to match the TIC order in the main data
            self.meta_df = self.meta_df.set_index('TIC').loc[self.tic_ids].reset_index()

            # Remove TIC and label columns from metadata
            self.meta_data = self.meta_df.drop(columns=['TIC', 'label']).values.astype(np.float32)
        else:
            self.meta_df = None
            self.meta_data = None

    def encode_labels(self):
        label_encoder = LabelEncoder()
        self.encoded_labels = label_encoder.fit_transform(self.labels)

        self.label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
        print("Label Mapping:")
        for label, index in self.label_mapping.items():
            print(f"'{index}' -> {label}")

    def split_data_by_tic(self):
        df = pd.DataFrame({'TIC': self.tic_ids, 'label': self.encoded_labels})
        tic_to_label = df.groupby('TIC')['label'].agg(lambda x: x.mode()[0]).reset_index()

        train_tics, val_tics = train_test_split(
            tic_to_label['TIC'],
            test_size=0.3,
            stratify=tic_to_label['label'],
            random_state=42
        )

        train_tic_set = set(train_tics)
        val_tic_set = set(val_tics)

        mask_train = [tic in train_tic_set for tic in self.tic_ids]
        mask_val = [tic in val_tic_set for tic in self.tic_ids]

        self.X_train = self.data_frame.values[mask_train].astype(np.float32)
        self.y_train = np.array(self.encoded_labels)[mask_train]
        self.tic_train = np.array(self.tic_ids)[mask_train]

        self.X_val = self.data_frame.values[mask_val].astype(np.float32)
        self.y_val = np.array(self.encoded_labels)[mask_val]
        self.tic_val = np.array(self.tic_ids)[mask_val]

        if self.meta_data is not None:
            self.train_meta = self.meta_data[mask_train].astype(np.float32)
            self.val_meta = self.meta_data[mask_val].astype(np.float32)

    def to_tensors(self):
        self.input_size = self.X_train.shape[1]

        # Shuffle training
        train_indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(train_indices)
        self.X_train = self.X_train[train_indices]
        self.y_train = self.y_train[train_indices]
        self.tic_train = self.tic_train[train_indices]
        if self.meta_data is not None:
            self.train_meta = self.train_meta[train_indices]

        # Shuffle validation
        val_indices = np.arange(self.X_val.shape[0])
        np.random.shuffle(val_indices)
        self.X_val = self.X_val[val_indices]
        self.y_val = self.y_val[val_indices]
        self.tic_val = self.tic_val[val_indices]
        if self.meta_data is not None:
            self.val_meta = self.val_meta[val_indices]

        # Convert to tensors
        self.train_data_tensor = torch.tensor(self.X_train, dtype=torch.float32).unsqueeze(1)
        self.train_label_tensor = torch.tensor(self.y_train, dtype=torch.long)
        self.val_data_tensor = torch.tensor(self.X_val, dtype=torch.float32).unsqueeze(1)
        self.val_label_tensor = torch.tensor(self.y_val, dtype=torch.long)

        if self.meta_data is not None:
            self.train_meta_tensor = torch.tensor(self.train_meta, dtype=torch.float32)
            self.val_meta_tensor = torch.tensor(self.val_meta, dtype=torch.float32)

        self.print_summary()
        print("Data processing completed.")

    def get_train_set(self):
        return self.train_data_tensor, self.train_label_tensor, self.tic_train

    def get_validation_set(self):
        return self.val_data_tensor, self.val_label_tensor, self.tic_val

    def get_train_meta(self):
        return self.train_meta_tensor if hasattr(self, 'train_meta_tensor') else None

    def get_val_meta(self):
        return self.val_meta_tensor if hasattr(self, 'val_meta_tensor') else None

    def get_label_mapping(self):
        return self.label_mapping

    def get_input_size(self):
        return self.input_size

    def get_orig_data(self):
        return self.data_orig
    
    def get_metadata_dim(self):
        """Returns the dimension of the metadata if available, else returns 0."""
        return self.meta_data.shape[1] if self.meta_data is not None else 0

    def print_summary(self):
        print(f"Input shape: {self.input_shape}")
        print(f"Train set size: {self.X_train.shape[0]}")
        print(f"Validation set size: {self.X_val.shape[0]}")
        print(f"Number of classes: {len(self.label_mapping)}")

