# -*- coding: utf-8 -*-
import argparse

import torch

from tqdm import tqdm

from models.embedding import ProtoNetEmbedding
from models.bilevel_head import R2D2Head, SVMHead

from utilities import set_gpu, count_accuracy, log, setup_seed
import numpy as np
import os
from dataloaders.tieredImageNet import tieredImageNet, FewShotDataloader


def get_model(opt):
    network = ProtoNetEmbedding().cuda()
    if opt.cls_head == 'R2D2':
        cls_head = R2D2Head().cuda()
    elif opt.cls_head == 'SVM':
        cls_head = SVMHead.cuda()

    return (network, cls_head)


if __name__ == '__main__':
    setup_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls-head', type=str, default='R2D2',
                        help='R2D2 or SVM')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--load', default='/data/liuyong/TianPinzhuo/IJCAI/r2d2_tieredImageNet_5way_batch4/best_model.pth.tar',
                        help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=2000,
                        help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                        help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=5,
                        help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=10,
                        help='number of query examples per training class')

    opt = parser.parse_args()
    set_gpu(opt.gpu)

    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model()

    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()
    #-----------------------------
    dataset_test = tieredImageNet(phase='test')
    data_loader = FewShotDataloader
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )
    #-----------------------------------

    # Evaluate on test set
    test_accuracies = []

    for i, batch in enumerate(tqdm(dloader_test()), 1):

        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        data_support = data_support.float()
        data_query = data_query.float()

        labels_support = labels_support.long()
        labels_query = labels_query.long()


        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query



        emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
        emb_support = emb_support.reshape(1, n_support, -1)

        emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(1, n_query, -1)

        logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)' \
                  .format(i, opt.episode, avg, ci95, acc))
