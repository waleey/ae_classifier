import torch
from torch import nn

class MLP_classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.negative_slope = 0.1
        self.dropout_p = 0.4

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=kwargs['in_shape'], out_features=kwargs['dense_shape'], bias=True),
            nn.BatchNorm1d(num_features=kwargs['dense_shape']),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_p)  # Dropout only here
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_features=kwargs['dense_shape'], out_features=kwargs['out_shape'], bias=True),
            nn.BatchNorm1d(num_features=kwargs['out_shape']),
            nn.LeakyReLU(negative_slope=self.negative_slope)
            # No dropout here
        )

    def forward(self, embedding):
        embedding = self.dense1(embedding)
        embedding = self.dense2(embedding)
        return embedding
