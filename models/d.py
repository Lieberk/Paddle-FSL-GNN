import sys

sys.path.append('..')
from paddle import nn


class Discriminator(nn.Layer):

    def __init__(self, hidden_size=230, num_labels=2):
        nn.Layer.__init__(self)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits
