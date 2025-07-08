"""
Cell combining classification model
with bens autoencoder model
"""
import network_models_1024 as nets_1024
import network_models_256 as nets_256
import network_models_128 as nets_128
import mlp_classifier as mlp
import torch.nn as nn

class Autoencoder_Classifier(nn.Module):
    def __init__(self, classifier_type='composite classifier', **kwargs):
        super().__init__()
        self.classifier_type = classifier_type.lower()
        self.load_model_dict()

        self.ae_model = self.ae_model_dict[kwargs.get('ae_model', 'nets_1024')]
        self.classifier_model = self.classifer_model_dict[kwargs.get('classifier_model', 'mlp')]

        self.encoder = self.ae_model.bens_cubic_code(input_shape=1)
        encoder_output_shape = self.encoder.get_output_shape()

        self.classifier = self.classifier_model.MLP_classifier(
            in_shape=encoder_output_shape,
            dense_shape=encoder_output_shape // 2,
            out_shape=kwargs['output_shape']
        )

        if self.classifier_type == 'composite classifier':
            self.decoder = nets_1024.bens_cubic_decode(input_shape=1)
            print(f"selected decoder: {self.decoder}")
        else:
            self.decoder = None

        print(f"selected encoder: {self.encoder}")
        print(f"selected classifier: {self.classifier}")

    def forward(self, feature):
        bottleneck = self.encoder(feature)
        class_logits = self.classifier(bottleneck)

        if self.classifier_type == 'composite classifier':
            reconstructed = self.decoder(bottleneck)
            return reconstructed, class_logits
        else:
            return class_logits  # no reconstruction

    def encdoer_fun(self, x):
        return self.encoder(x)

    def decoder_fun(self, x):
        if self.decoder is None:
            raise RuntimeError("Decoder is not initialized in 'direct classifier' mode.")
        return self.decoder(x)

    def reconstruct_fun(self, x):
        if self.decoder is None:
            raise RuntimeError("Decoder is not initialized in 'direct classifier' mode.")
        return self.decoder(self.encoder(x))

    def load_model_dict(self):
        self.ae_model_dict = {
            'nets_1024': nets_1024,
            'nets_256': nets_256,
            'nets_128': nets_128
        }
        self.classifer_model_dict = {
            'mlp': mlp
        }
