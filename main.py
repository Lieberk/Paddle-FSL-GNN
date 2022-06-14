import os
import argparse
import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
from data import generator
from utils import io_utils
import models.models as models
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
parser.add_argument('--exp_name', type=str, default='minimagenet_N5_S1', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--iterations', type=int, default=50000, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
                    help='Learning rate decay interval')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', default=True, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=10000, metavar='N',
                    help='how many batches between each model saving')
parser.add_argument('--test_interval', type=int, default=2000, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--test_N_way', type=int, default=5, metavar='N',
                    help='Number of classes for doing each classification run')
parser.add_argument('--train_N_way', type=int, default=5, metavar='N',
                    help='Number of classes for doing each training comparison')
parser.add_argument('--test_N_shots', type=int, default=1, metavar='N',
                    help='Number of shots in test')
parser.add_argument('--train_N_shots', type=int, default=1, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--metric_network', type=str, default='gnn_iclr_nl', metavar='N',
                    help='gnn_iclr_nl')
parser.add_argument('--active_random', type=int, default=0, metavar='N',
                    help='random active ? ')
parser.add_argument('--dataset_root', type=str, default='pretrain', metavar='N',
                    help='Root dataset')
parser.add_argument('--test_samples', type=int, default=10000, metavar='N',
                    help='Number of shots')
parser.add_argument('--dataset', type=str, default='miniImagenet', metavar='N',
                    help='miniImagenet')
parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                    help='Decreasing the learning rate every x iterations')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--N', default=5, type=int,
                    help='N way')
parser.add_argument('--K', default=5, type=int,
                    help='K shot')
args = parser.parse_args()

args.train_N_way = args.N
args.train_N_shots = args.K
args.test_N_way = args.N
args.test_N_shots = args.K

if args.K == 1:
    args.batch_size = 200
    args.iterations = 15000
elif args.K == 5:
    args.batch_size = 40
    args.iterations = 90000


def _init_():
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('checkpoint/' + args.exp_name):
        os.makedirs('checkpoint/' + args.exp_name)
    if not os.path.exists('checkpoint/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoint/' + args.exp_name + '/' + 'models')


_init_()

io = io_utils.IOStream('checkpoint/' + args.exp_name + '/run.log')
io.cprint(str(args))
np.random.seed(args.seed)


def test_one_shot(args, model, test_samples=5000, partition='test'):
    io = io_utils.IOStream('checkpoint/' + args.exp_name + '/run.log')

    io.cprint('\n**** TESTING WITH %s ***' % (partition,))

    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples / args.batch_size_test)
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra)
        [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, hidden_labels] = data

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            hidden_labels = hidden_labels.cuda()
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [batch_xi for batch_xi in xi_s]
        labels_yi = [label_yi for label_yi in labels_yi]
        oracles_yi = [oracle_yi for oracle_yi in oracles_yi]

        # Compute embedding from x and xi_s
        z = enc_nn(x)[-1]
        zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

        # Compute metric from embeddings
        output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
        output = out_logits
        y_pred = softmax_module.forward(output)
        y_pred = y_pred.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        labels_x_cpu = labels_x_cpu.numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)

        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1

        if (i + 1) % 100 == 0:
            io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0 * correct / total))

    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0 * correct / total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
    enc_nn.train()
    metric_nn.train()

    return 100.0 * correct / total


def train_batch(model, data):
    [enc_nn, metric_nn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi, hidden_labels] = data

    # Compute embedding from x and xi_s
    z = enc_nn(batch_x)[-1]
    zi_s = [enc_nn(batch_xi)[-1] for batch_xi in batches_xi]

    # Compute metric from embeddings
    out_metric, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
    logsoft_prob = softmax_module.forward(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = paddle.to_tensor(formatted_label_x).cast('int64')
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()

    return loss


def train():
    train_loader = generator.Generator(args.dataset_root, args, partition='train', dataset=args.dataset)
    io.cprint('Batch size: ' + str(args.batch_size))

    # Try to load models
    enc_nn = models.load_model('enc_nn', args, io)
    metric_nn = models.load_model('metric_nn', args, io)

    if enc_nn is None or metric_nn is None:
        enc_nn, metric_nn = models.create_models(args=args)
    softmax_module = models.SoftmaxModule()

    io.cprint(str(enc_nn))
    io.cprint(str(metric_nn))

    weight_decay = 0
    if args.dataset == 'miniImagenet':
        print('Weight decay ' + str(1e-6))
        weight_decay = 1e-6

    opt_enc_nn = optim.Adam(parameters=enc_nn.parameters(),
                            learning_rate=args.lr, weight_decay=weight_decay)
    opt_metric_nn = optim.Adam(parameters=metric_nn.parameters(),
                               learning_rate=args.lr, weight_decay=weight_decay)

    enc_nn.train()
    metric_nn.train()
    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    for batch_idx in range(args.iterations):

        ####################
        # Train
        ####################
        data = train_loader.get_task_batch(batch_size=args.batch_size, n_way=args.train_N_way,
                                           unlabeled_extra=args.unlabeled_extra, num_shots=args.train_N_shots)
        [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi, hidden_labels] = data

        opt_enc_nn.clear_grad()
        opt_metric_nn.clear_grad()

        loss_d_metric = train_batch(model=[enc_nn, metric_nn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi, hidden_labels])

        opt_enc_nn.step()
        opt_metric_nn.step()

        adjust_learning_rate(optimizers=[opt_enc_nn, opt_metric_nn], lr=args.lr, iter=batch_idx)

        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss_d_metric.item()
        if batch_idx % args.log_interval == 0:
            display_str = 'Train Iter: {}'.format(batch_idx)
            display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss / counter)
            io.cprint(display_str)
            counter = 0
            total_loss = 0

        ####################
        # Val
        ####################
        if (batch_idx + 1) % args.test_interval == 0 or batch_idx == 20:
            if batch_idx == 20:
                test_samples = 100
            else:
                test_samples = 3000
            if args.dataset == 'miniImagenet':
                val_acc_aux = test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                                            test_samples=test_samples * 5, partition='val')

            if val_acc_aux is not None and val_acc_aux >= val_acc:
                val_acc = val_acc_aux

            if args.dataset == 'miniImagenet':
                io.cprint("Best test accuracy {:.4f} \n".format(val_acc))

        ####################
        # Save model
        ####################
        if (batch_idx + 1) % args.save_interval == 0:
            paddle.save(enc_nn.state_dict(), 'checkpoint/%s/models/enc_nn.pdparams' % args.exp_name)
            paddle.save(metric_nn.state_dict(), 'checkpoint/%s/models/metric_nn.pdparams' % args.exp_name)


def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        optimizer.set_lr = new_lr


if __name__ == "__main__":
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        enc_nn, metric_nn = models.create_models(args=args)
        enc_nn_state_dict = models.load_model('enc_nn', args, io)
        metric_nn_state_dict = models.load_model('metric_nn', args, io)
        enc_nn.load_dict(enc_nn_state_dict)
        metric_nn.load_dict(metric_nn_state_dict)
        softmax_module = models.SoftmaxModule()
        test_one_shot(args, model=[enc_nn, metric_nn, softmax_module],
                      test_samples=args.test_samples)