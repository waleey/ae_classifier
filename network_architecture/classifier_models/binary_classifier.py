import torch
from torch import nn

class Binary_classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.negative_slope = 0.1
        self.dropout_p = 0.2

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=kwargs['in_shape'], out_features=kwargs['dense_shape'], bias=True),
            nn.BatchNorm1d(num_features=kwargs['dense_shape']),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_p)
        )

        self.output_layer = nn.Linear(
            in_features=kwargs['dense_shape'], out_features=1, bias=True
        )
        # No BatchNorm or activation here (required for BCEWithLogitsLoss)

    def forward(self, embedding):
        x = self.dense1(embedding)
        x = self.output_layer(x)  # shape: (N, 1)
        return x.squeeze(1)       # shape: (N,)
