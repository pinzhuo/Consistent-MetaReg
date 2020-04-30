# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models.embedding import ProtoNetEmbedding
from dataloaders.office31 import Office31Dataset

from protoNet.prototy_head import ClassificationHead
from utilities import set_gpu, count_accuracy, Timer, check_dir, log, setup_seed


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def get_model():
    network = ProtoNetEmbedding().cuda()
    cls_head = ClassificationHead(enable_scale=False).cuda()

    return (network, cls_head)


if __name__ == '__main__':
    setup_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=2,
                        help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=5,
                        help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                        help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=5,
                        help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=500,
                        help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=10,
                        help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                        help='number of classes in one training episode')

    parser.add_argument('--save-path', default='/data/save_models/...')
    parser.add_argument('--gpu', default='0')

    parser.add_argument('--episodes-per-batch', type=int, default=4,
                        help='number of episodes per batch')

    opt = parser.parse_args()

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model()

    optimizer = torch.optim.Adam([{'params': embedding_net.parameters()},
                                  {'params': cls_head.parameters()}], lr=0.001)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    max_val_acc = 0.0

    x_entropy = torch.nn.CrossEntropyLoss()

    timer = Timer()

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        #scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []

        train_dataset = Office31Dataset(num_classes=opt.train_way, num_support=opt.train_shot,
                                      num_query=opt.train_query, num_epoch=1000, phase='train')
        dloader_train = DataLoader(train_dataset, shuffle=True, num_workers=32, batch_size=opt.episodes_per_batch)


        for i, batch in enumerate(tqdm(dloader_train)):

            data_support, _, labels_support, data_query, _, labels_query = [x.cuda() for x in batch]

            data_support = data_support.float()
            data_query = data_query.float()

            labels_support = labels_support.long()
            labels_query = labels_query.long()

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 50 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path,
                    'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                        epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        dloader_val = Office31Dataset(num_classes=opt.train_way, num_support=opt.val_shot,
                                    num_query=opt.val_query, num_epoch=opt.val_episode, phase='val')
        dloader_val = DataLoader(dloader_val, shuffle=True, num_workers=32, batch_size=1)

        for i, batch in enumerate(tqdm(dloader_val)):
            data_support, _, labels_support, data_query, _, labels_query = [x.cuda() for x in batch]

            data_support = data_support.float()
            data_query = data_query.float()

            labels_support = labels_support.long()
            labels_query = labels_query.long()

            test_n_support = opt.train_way * opt.val_shot
            test_n_query = opt.train_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg

            state = {'epoch': epoch + 1, 'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(),
                     'optimizer':optimizer.state_dict()}
            torch.save(state
                       , os.path.join(opt.save_path, 'best_model.pth.tar'.format(epoch)))

            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        if epoch % opt.save_epoch == 0:

            state = {'epoch': epoch + 1, 'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(),
                     'optimizer':optimizer.state_dict()}
            torch.save(state
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))