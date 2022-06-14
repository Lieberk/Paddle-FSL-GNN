import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models import pd_api


class Encoder(nn.Layer):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Layer.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1D(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1D(max_length)

        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.set_value(
            paddle.to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).cast('float32'))
        self.mask_embedding.weight.stop_gradient = True
        self._minus = -100

    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(pd_api.transpose(inputs, 1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2)  # n x hidden_size

    def pcnn(self, inputs, mask):
        x = self.conv(pd_api.transpose(inputs, 1, 2))  # n x hidden x length
        mask = 1 - pd_api.transpose(self.mask_embedding(mask), 1, 2)  # n x 3 x length
        pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
        pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
        x = paddle.concat([pool1, pool2, pool3], 1)
        x = x.squeeze(2)  # n x (hidden_size * 3)
