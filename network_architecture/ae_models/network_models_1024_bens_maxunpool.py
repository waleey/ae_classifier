import torch
from torch import nn
import torch.nn.functional as F

#defining global variables 
global_size_depool1 = None
global_size_depool2 = None
global_size_depool3 = None
global_size_depool4 = None
global_idx1 = None
global_idx2 = None
global_idx3 = None
global_idx4 = None

class bens_maxunpool_1024(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #defines kernel and scale factor size centrally
        self.kernel_0 = 1
        self.kernel_1 = 5
        self.kernel_2 = 5
        self.kernel_3 = 3
        self.kernel_4 = 3
        self.kernel_5 = 3
        self.kernel_6 = 3
        self.kernel_7 = 3
        self.kernel_8 = 3
        self.kernel_9 = 1
        self.kernel_10 = 1
        self.kernel_11 = 1
        self.kernel_12 = 1
        self.kernel_13 = 1
        self.kernel_14 = 1
        self.kernel_15 = 1
        self.pool_1 = 4
        self.pool_2 = 2
        self.pool_3 = 2
        self.pool_4 = 2
        self.dense_1_in = 32
        self.dense_1_out = 32
        self.negative_slope = 0.01
        self.dropout = 0.4

        #saving the maxpool idx
        self.idx1 = None
        self.idx2 = None
        self.idx3 = None
        self.idx4 = None

        #saving depool shape
        self.size_depool1 = None
        self.size_depool2 = None
        self.size_depool3 = None
        self.size_depool4 = None

        #required extra functions
        self.unflatten = nn.Unflatten(1,(1,32))

        self.leakyrelu = nn.LeakyReLU(negative_slope = 0.1)

        """
        re-writing the convolutional layers
        in modular form

        Format:
        Convolutinal Layer
        Batch Normalization
        Activation
        """

        self.en_layer1 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_1, stride=1, padding = self.kernel_1//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer2 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_2, stride=1, padding = self.kernel_2//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer3 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_3, stride=1, padding = self.kernel_3//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer4 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_4, stride=1, padding = self.kernel_4//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer5 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_5, stride=1, padding = self.kernel_5//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer6 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_6, stride=1, padding = self.kernel_6//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer7 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_7, stride=1, padding = self.kernel_7//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer8 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_8, stride=1, padding = self.kernel_8//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer9 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_9, stride=1, padding = self.kernel_9//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer10 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_10, stride=1, padding = self.kernel_10//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer11 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_11, stride=1, padding = self.kernel_11//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer12 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_12, stride=1, padding = self.kernel_12//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer13 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_13, stride=1, padding = self.kernel_13//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.en_layer14 = nn.Sequential(
            nn.Conv1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_14, stride=1, padding = self.kernel_14//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )

        self.en_pool_1 = nn.MaxPool1d(
            kernel_size=self.pool_1, return_indices=True
        )

        self.en_pool_2 = nn.MaxPool1d(
            kernel_size=self.pool_2, return_indices=True
        )

        self.en_pool_3 = nn.MaxPool1d(
            kernel_size=self.pool_3, return_indices=True
        )

        self.en_pool_4 = nn.MaxPool1d(
            kernel_size=self.pool_4, return_indices=True
        )

        self.en_dropout = nn.Dropout(p=self.dropout)

        self.en_dense_1 = nn.Linear(
            in_features = self.dense_1_in, out_features = self.dense_1_out
        )

        """
        Defining all decoder layers here
        in a more modular format
        """

        self.de_layer1 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_1, stride=1, padding = self.kernel_1//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer2 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_2, stride=1, padding = self.kernel_2//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer3 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_3, stride=1, padding = self.kernel_3//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer4 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_4, stride=1, padding = self.kernel_4//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer5 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_5, stride=1, padding = self.kernel_5//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer6 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_6, stride=1, padding = self.kernel_6//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer7 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_7, stride=1, padding = self.kernel_7//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer8 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_8, stride=1, padding = self.kernel_8//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer9 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_9, stride=1, padding = self.kernel_9//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer10 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_10, stride=1, padding = self.kernel_10//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer11 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_11, stride=1, padding = self.kernel_11//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer12 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_12, stride=1, padding = self.kernel_12//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer13 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_13, stride=1, padding = self.kernel_13//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )
        self.de_layer14 = nn.Sequential(
            nn.ConvTranspose1d(
            in_channels=kwargs["input_shape"], out_channels=kwargs["input_shape"],
            kernel_size=self.kernel_14, stride=1, padding = self.kernel_14//2),
            nn.BatchNorm1d(num_features=kwargs["input_shape"]),
            nn.LeakyReLU(negative_slope = self.negative_slope)
        )

        self.de_pool_1 = nn.MaxUnpool1d(
            kernel_size=self.pool_1
        )

        self.de_pool_2 = nn.MaxUnpool1d(
            kernel_size=self.pool_2
        )

        self.de_pool_3 = nn.MaxUnpool1d(
            kernel_size=self.pool_3
        )

        self.de_pool_4 = nn.MaxUnpool1d(
            kernel_size=self.pool_4
        )

        self.de_dropout = nn.Dropout(p=self.dropout)

        self.de_dense_1 = nn.Linear(
            in_features = self.dense_1_out, out_features = self.dense_1_in
        )

        #batch normalization layers
        self.en_dense_nor_1 = nn.BatchNorm1d(
            num_features=self.dense_1_out
        )
    def forward(self,features):
        """
        Temporarily writing the modular
        forward functions
        """
        # Encoder
        global global_size_depool1, global_size_depool2, global_size_depool3, global_size_depool4
        activation = self.en_layer1(features)
        global_size_depool1 = activation
        activation = self.en_layer2(features)

        activation, self.idx1 = self.en_pool_1(activation)
        global_size_depool2 = activation

        activation = self.en_layer3(activation)
        activation = self.en_layer4(activation)
        activation = self.en_layer5(activation)

        activation, self.idx2 = self.en_pool_2(activation)
        global_size_depool3 = activation

        activation = self.en_layer6(activation)
        activation = self.en_layer7(activation)

        activation, self.idx3 = self.en_pool_3(activation)
        global_size_depool4 = activation

        activation = self.en_layer8(activation)
        activation = self.en_layer9(activation)

        activation, self.idx4 = self.en_pool_4(activation)

        activation = self.en_layer10(activation)
        activation = self.en_layer11(activation)
        activation = self.en_layer12(activation)
        activation = self.en_layer13(activation)
        activation = self.en_layer14(activation)

        activation = torch.flatten(activation,start_dim=1)
        activation = self.en_dense_1(activation)

        #Decoder
        activation = self.de_dense_1(activation)
        activation = self.unflatten(activation)
        activation = self.de_layer14(activation)
        activation = self.de_layer13(activation)
        activation = self.de_layer12(activation)
        activation = self.de_layer11(activation)
        activation = self.de_layer10(activation)

        activation = self.de_pool_4(activation, self.idx4, output_size = global_size_depool4.size())

        activation = self.de_layer9(activation)
        activation = self.de_layer8(activation)

        activation = self.de_pool_3(activation, self.idx3, output_size = self.size_depool3.size())

        activation = self.de_layer7(activation)
        activation = self.de_layer6(activation)

        activation = self.de_pool_2(activation, self.idx2, output_size = self.size_depool2.size())

        activation = self.de_layer5(activation)
        activation = self.de_layer4(activation)
        activation = self.de_layer3(activation)

        activation = self.de_pool_1(activation, self.idx1, output_size = self.size_depool1.size())

        activation = self.de_layer2(activation)
        activation = self.de_layer1(activation)
        reconstructed = activation
        return reconstructed
    def get_output_shape(self):
        return self.dense_1_out

class encode(bens_maxunpool_1024):

    def forward(self,features):
        global global_size_depool1, global_size_depool2, global_size_depool3, global_size_depool4, global_idx1, global_idx2, global_idx3, global_idx4
        activation = self.en_layer1(features)
        global_size_depool1 = activation
        activation = self.en_layer2(activation)

        activation, global_idx1 = self.en_pool_1(activation)
        global_size_depool2 = activation

        activation = self.en_layer3(activation)
        activation = self.en_layer4(activation)
        activation = self.en_layer5(activation)

        activation, global_idx2 = self.en_pool_2(activation)
        global_size_depool3 = activation

        activation = self.en_layer6(activation)
        activation = self.en_layer7(activation)

        activation, global_idx3 = self.en_pool_3(activation)
        global_size_depool4 = activation

        activation = self.en_layer8(activation)
        activation = self.en_layer9(activation)

        activation, global_idx4 = self.en_pool_4(activation)

        activation = self.en_layer10(activation)
        activation = self.en_layer11(activation)
        activation = self.en_dropout(activation)
        activation = self.en_layer12(activation)
        activation = self.en_layer13(activation)
        activation = self.en_layer14(activation)

        activation = torch.flatten(activation,start_dim=1)
        activation = self.en_dense_1(activation)
        activation = F.leaky_relu(self.en_dense_1(activation), negative_slope=self.negative_slope)
        activation = self.en_dropout(activation)

        code = activation

        return code
    def get_output_shape(self):
        return super().get_output_shape()
    
class decode(bens_maxunpool_1024):
    def forward(self, features):
        global global_size_depool1, global_size_depool2, global_size_depool3, global_size_depool4, global_idx1, global_idx2, global_idx3, global_idx4
        activation = self.de_dense_1(features)
        activation = F.leaky_relu(self.en_dense_1(activation), negative_slope=self.negative_slope)
        activation = self.de_dropout(activation)
        activation = self.unflatten(activation)

        activation = self.de_layer14(activation)
        activation = self.de_layer13(activation)
        activation = self.de_layer12(activation)
        activation = self.de_dropout(activation)
        activation = self.de_layer11(activation)
        activation = self.de_layer10(activation)

        activation = self.de_pool_4(activation, global_idx4, output_size=global_size_depool4.size())

        activation = self.de_layer9(activation)
        activation = self.de_layer8(activation)

        activation = self.de_pool_3(activation, global_idx3, output_size=global_size_depool3.size())

        activation = self.de_layer7(activation)
        activation = self.de_layer6(activation)

        activation = self.de_pool_2(activation, global_idx2, output_size=global_size_depool2.size())

        activation = self.de_layer5(activation)
        activation = self.de_layer4(activation)
        activation = self.de_layer3(activation)

        activation = self.de_pool_1(activation, global_idx1, output_size=global_size_depool1.size())

        activation = self.de_layer2(activation)
        activation = self.de_layer1(activation)

        reconstructed = activation
        del global_size_depool1, global_size_depool2, global_size_depool3, global_size_depool4, global_idx1, global_idx2, global_idx3, global_idx4
        return reconstructed
