import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models import gnn_iclr


class EmbeddingImagenet(nn.Layer):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, args, emb_size):
        super(EmbeddingImagenet, self).__init__()
        self.emb_size = emb_size
        self.ndf = 64
        self.args = args

        # Input 84x84x3
        self.conv1 = nn.Conv2D(3, self.ndf, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2D(self.ndf, int(self.ndf*1.5), kernel_size=3, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(int(self.ndf*1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2D(int(self.ndf*1.5), self.ndf*2, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.ndf*2)
        self.drop_3 = nn.Dropout2D(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2D(self.ndf*2, self.ndf*4, kernel_size=3, padding=1, bias_attr=False)
        self.bn4 = nn.BatchNorm2D(self.ndf*4)
        self.drop_4 = nn.Dropout2D(0.5)

        # Input 5x5x256
        self.fc1 = nn.Linear(self.ndf*4*5*5, self.emb_size, bias_attr=True)
        self.bn_fc = nn.BatchNorm1D(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2)
        x = self.drop_4(x)
        x = x.reshape([-1, self.ndf*4*5*5])
        output = self.bn_fc(self.fc1(x))

        return [e1, e2, e3, e4, None, output]


class MetricNN(nn.Layer):
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = emb_size
        self.args = args

        if self.metric_network == 'gnn_iclr_nl':
            assert(self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            if self.args.dataset == 'miniImagenet':
                self.gnn_obj = gnn_iclr.GNN_nl(args.train_N_way, num_inputs, nf=96, J=1)
        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z, zi_s, labels_yi):
        # Creating WW matrix
        zero_pad = paddle.zeros(labels_yi[0].shape)
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [paddle.concat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = paddle.concat(nodes, 1)

        logits = self.gnn_obj(nodes).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits

    def gnn_iclr_active_forward(self, z, zi_s, labels_yi, oracles_yi, hidden_layers):
        # Creating WW matrix
        zero_pad = paddle.zeros(labels_yi[0].shape)
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [paddle.concat([label_yi, zi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = paddle.concat(nodes, 1)

        oracles_yi = [zero_pad] + oracles_yi
        oracles_yi = [oracle_yi.unsqueeze(1) for oracle_yi in oracles_yi]
        oracles_yi = paddle.concat(oracles_yi, 1)

        logits = self.gnn_obj(nodes, oracles_yi, hidden_layers).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits

    def forward(self, inputs):
        '''input: [batch_x, [batches_xi], [labels_yi]]'''
        [z, zi_s, labels_yi, oracles_yi, hidden_labels] = inputs

        if 'gnn_iclr_active' in self.metric_network:
           return self.gnn_iclr_active_forward(z, zi_s, labels_yi, oracles_yi, hidden_labels)
        elif 'gnn_iclr' in self.metric_network:
            return self.gnn_iclr_forward(z, zi_s, labels_yi)
        else:
            raise NotImplementedError


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise(NotImplementedError)


def load_model(model_name, args, io):
    try:
        model = paddle.load('checkpoint/%s/models/%s.pdparams' % (args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None


def create_models(args):
    print(args.dataset)
    if 'miniImagenet' == args.dataset:
        enc_nn = EmbeddingImagenet(args, 128)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn, MetricNN(args, emb_size=enc_nn.emb_size)
