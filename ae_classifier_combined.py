import network_models_1024 as nets_1024
import network_models_256 as nets_256
import network_models_128 as nets_128
import network_models_1024_bens_transpose as nets_1024_bens_transpose
import network_models_1024_bens_maxunpool as nets_1024_bens_maxunpool
from mlp_classifier import MLP_classifier as mlp
from binary_classifier import Binary_classifier as bc
import torch.nn as nn
import torch


class autoencoder_classifier(nn.Module):
    def __init__(self, metadata_dim=0, **kwargs):
        super().__init__()
        self.load_model_dict()

        ae_model_name = kwargs.get('ae_model', 'nets_1024')
        classifier_model_name = kwargs.get('classifier_model', 'mlp')
        output_shape = kwargs.get('output_shape', 2)
        self.metadata_dim = metadata_dim

        self.ae_model = self.ae_model_dict[ae_model_name]
        self.classifier_model_class = self.classifier_model_dict[classifier_model_name]

        self.encoder = self.ae_model.encode(input_shape=1)
        self.decoder = self.ae_model.decode(input_shape=1)

        self.encoder_output_shape = self.encoder.get_output_shape()
        total_input_dim = self.encoder_output_shape + metadata_dim

        self.classifier = self.classifier_model_class(
            in_shape=total_input_dim,
            dense_shape=total_input_dim // 2,
            out_shape=output_shape
        )

        self.train_metadata = None
        self.val_metadata = None

        print(f"selected encoder: {self.encoder}")
        print(f"selected decoder: {self.decoder}")
        print(f"selected classifier: {self.classifier}")

    def set_train_metadata(self, train_meta_tensor=None):
        self.train_metadata = train_meta_tensor

    def set_val_metadata(self, val_meta_tensor=None):
        self.val_metadata = val_meta_tensor


    def forward(self, feature):
        bottleneck = self.encoder(feature)
        reconstructed = self.decoder(bottleneck)

        if self.metadata_dim > 0:
            if self.training:
                if self.train_metadata is None:
                    raise ValueError("Metadata expected, but train_metadata not set.")
                meta = self.train_metadata.to(feature.device)
            else:
                if self.val_metadata is None:
                    raise ValueError("Metadata expected, but val_metadata not set.")
                meta = self.val_metadata.to(feature.device)

            if meta.shape[0] != bottleneck.shape[0]:
                raise ValueError(f"Metadata batch size {meta.shape[0]} doesn't match bottleneck batch size {bottleneck.shape[0]}")

            bottleneck = torch.cat([bottleneck, meta], dim=1)

        class_logits = self.classifier(bottleneck)
        return reconstructed, class_logits

    def encdoer_fun(self, x):
        return self.encoder(x)

    def decoder_fun(self, x):
        return self.decoder(x)

    def reconstruct_fun(self, x):
        return self.decoder(self.encoder(x))

    def load_model_dict(self):
        self.ae_model_dict = {
            'nets_1024': nets_1024,
            'nets_256': nets_256,
            'nets_128': nets_128,
            'nets_1024_bens_transpose': nets_1024_bens_transpose,
            'nets_1024_bens_maxunpool': nets_1024_bens_maxunpool
        }
        self.classifier_model_dict = {
            'mlp': mlp,
            'binary_classifier': bc
        }
