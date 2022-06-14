import sys

sys.path.append('..')
import fewshot_re_kit
import paddle

from . import gnn_iclr


class GNN(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, N, hidden_size=230):
        '''
        N: Num of classes
        '''
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.node_dim = hidden_size + N
        self.gnn_obj = gnn_iclr.GNN_nl(N, self.node_dim, nf=96, J=1)

    def forward(self, support, query, N, K, NQ):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)
        query = self.sentence_encoder(query)
        support = support.reshape([-1, N, K, self.hidden_size])
        query = query.reshape([-1, NQ, self.hidden_size])

        B = support.shape[0]
        D = self.hidden_size
        support = support.unsqueeze(1).expand([-1, NQ, -1, -1, -1]).reshape([-1, N * K, D])  # (B * NQ, N * K, D)
        query = query.reshape([-1, 1, D])  # (B * NQ, 1, D)
        labels = paddle.zeros((B * NQ, 1 + N * K, N), dtype='float32')
        for b in range(B * NQ):
            for i in range(N):
                for k in range(K):
                    labels[b, 1 + i * K + k, i] = 1
        nodes = paddle.concat([paddle.concat([query, support], 1), labels], -1)  # (B * NQ, 1 + N * K, D + N)

        logits = self.gnn_obj(nodes)  # (B * NQ, N)
        _, pred = paddle.topk(logits, 1)
        return logits, pred
