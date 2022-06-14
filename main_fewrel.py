from fewshot_re_kit.data_loader import get_loader, get_loader_pair
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, \
    RobertaPAIRSentenceEncoder
from models.gnn import GNN
import paddle.optimizer as optim
import json

import argparse
import numpy as np

import logging
import os


def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(filename=filename,
                        level=logging.INFO,
                        format=log_format,
                        datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
                        help='train file')
    parser.add_argument('--val', default='val_wiki',
                        help='val file')
    parser.add_argument('--test', default='val_wiki',
                        help='test file')
    parser.add_argument('--trainN', default=5, type=int,
                        help='N in train')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=10000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=1000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='gnn',
                        help='model name')
    parser.add_argument('--encoder', default='cnn',
                        help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--lr', default=-1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
                        help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--only_test', default=False, type=bool,
                        help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
                        help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = get_logger(
        filename=os.path.join('logs/', 'fewrel-{}way-{}shot.log'.format(N, K)),
        logger_name='master_logger')

    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
            glove_mat,
            glove_word2id,
            max_length)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                pretrain_ckpt,
                max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep,
                mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'
        if opt.pair:
            sentence_encoder = RobertaPAIRSentenceEncoder(
                pretrain_ckpt,
                max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep)
    else:
        raise NotImplementedError

    if opt.pair:
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                                            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                                          N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                                           N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
    else:
        train_data_loader = get_loader(opt.train, sentence_encoder,
                                       N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                                     N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                                      N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)

    if opt.optim == 'sgd':
        pp_optim = optim.SGD
    elif opt.optim == 'adam':
        pp_optim = optim.Adam
    elif opt.optim == 'adamw':
        import paddle.optimizer.AdamW as AdamW
        pp_optim = AdamW
    else:
        raise NotImplementedError

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    if model_name == 'gnn':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.zip'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                        pp_optim=pp_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        na_rate=opt.na_rate, val_step=opt.val_step, pair=opt.pair,
                        train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim,
                        learning_rate=opt.lr, grad_iter=opt.grad_iter, logger=logger)
    else:
        acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
        print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
