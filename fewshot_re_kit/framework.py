import paddle
from paddle import nn
import paddle.optimizer as optim

import os
import sys


def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class FewShotREModel(nn.Layer):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Layer.__init__(self)
        super().__init__()
        self.sentence_encoder = paddle.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.shape[-1]
        return self.cost(logits.reshape([-1, N]), label.flatten())

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return paddle.mean((pred.flatten() == label.flatten()).cast('float32'))


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = paddle.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = paddle.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pp_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              pair=False,
              logger=None):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")

        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate, warmup_steps=warmup_step, start_lr=0, end_lr=0.5, verbose=False)
            optimizer = optim.AdamW(parameters=parameters_to_optimize,
                                    learning_rate=scheduler,
                                    grad_clip=paddle.nn.ClipGradByNorm(clip_norm=10.0))

        else:
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=learning_rate, step_size=lr_step_size, gamma=0.8, verbose=False)
            optimizer = pp_optim(parameters=model.parameters(),
                                 learning_rate=scheduler, weight_decay=weight_decay,
                                 grad_clip=paddle.nn.ClipGradByNorm(clip_norm=10.0))

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            if pair:
                batch, label = next(self.train_data_loader)
                logits, pred = model(batch, N_for_train, K,
                                     Q * N_for_train + na_rate * Q)
            else:
                support, query, label = next(self.train_data_loader)

                logits, pred = model(support, query,
                                     N_for_train, K, Q * N_for_train + na_rate * Q)
            loss = model.loss(logits, label.cast('int64')) / float(grad_iter)
            right = model.accuracy(pred, label)

            loss.backward()

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.clear_grad()

            iter_loss += self.item(loss)
            iter_right += self.item(right)
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                                        100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
            logger.info('[TRAIN] Iteration {:d} | loss {:f} | accuracy {:f}'.format(it, iter_loss / iter_sample,
                                                                            100 * iter_right / iter_sample))

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter,
                                na_rate=na_rate, pair=pair, logger=logger)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    paddle.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
             model,
             B, N, K, Q,
             eval_iter,
             na_rate=0,
             pair=False,
             ckpt=None,
             logger=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for name, param in state_dict.items():
                    new_state_dict[name] = param
                model.load_dict(new_state_dict)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with paddle.no_grad():
            for it in range(eval_iter):
                if pair:
                    batch, label = next(eval_dataset)
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                else:
                    support, query, label = next(eval_dataset)
                    logits, pred = model(support, query, N, K, Q * N + Q * na_rate)

                right = model.accuracy(pred, label)
                iter_right += self.item(right)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
                if logger:
                    logger.info('[EVAL] Iteration {:d} | accuracy {:f}'.format(it, 100 * iter_right / iter_sample))
            print("")
        return iter_right / iter_sample
