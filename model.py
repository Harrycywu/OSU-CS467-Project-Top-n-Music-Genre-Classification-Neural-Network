import torch.nn as nn
import torch.nn.functional as F


# Self Designed CNN Model
class CNNModel(nn.Module):
    def __init__(self, kernel_size, stride, dropout_rate, pooling_stride, padding):
        super(CNNModel, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.pooling_stride = pooling_stride
        self.padding = padding

        # Batch Normalization: input_shape=(3, 256, 256)
        self.batchnorm = nn.BatchNorm2d(3)

        # Convolution Layer 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.act1 = nn.ReLU()
        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.pooling_stride)

        # Convolution Layer 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.act2 = nn.ReLU()
        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.pooling_stride)

        # Convolution Layer 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.act3 = nn.ReLU()
        # Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=self.pooling_stride)

        # Dropout to avoid overfitting (Input comes from nn.Conv2d module)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        # Dense Layers
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(57600, 128)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(128, 64)

        # Fully Connected Layer 3
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Batch Normalization
        output = self.batchnorm(x)

        # Convolution Layer 1
        output = self.act1(self.cnn1(output))
        # Max Pooling 1
        output = self.maxpool1(output)

        # Convolution Layer 2
        output = self.act2(self.cnn2(output))
        # Max Pooling 2
        output = self.maxpool2(output)

        # Convolution Layer 3
        output = self.act3(self.cnn3(output))
        # Max Pooling 3
        output = self.maxpool3(output)

        # Dropout 1
        output = self.dropout1(output)

        # Flatten
        output = output.view(output.size(0), -1)

        # Dense Layers
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))

        # nn.CrossEntropyLoss needs to be the raw output of the network, not the output of the softmax function
        output = self.fc3(output)

        return output
