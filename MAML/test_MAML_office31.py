import torch, os
import numpy as np
from dataloaders.office31 import Office31Dataset
import scipy.stats
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utilities import setup_seed
from MAML.meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    setup_seed(1234)

    print(args)

    config = [
        ('conv2d', [64, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 64 * 5 * 5])
    ]

    #device = torch.device('cuda')
    maml = Meta(args, config)
    maml.net.load_state_dict(torch.load(args.load_path)['model'])

    # ---------------------------------------------------

    test_dataset = Office31Dataset(num_classes=args.n_way, num_support=args.k_spt,
                                    num_query=args.k_qry, num_epoch=args.epoch, phase='test')
    dloader_test = DataLoader(test_dataset, shuffle=True, num_workers=32, batch_size= 1)

    test_accuracies = []

    for i, batch in enumerate(tqdm(dloader_test)):

        data_support, _, labels_support, data_query, _, labels_query = [x.cuda() for x in batch]

        data_support = data_support.float().squeeze(0)
        data_query = data_query.float().squeeze(0)

        labels_support = labels_support.long().squeeze(0)
        labels_query = labels_query.long().squeeze(0)

        accs = maml.finetunning(data_support, labels_support, data_query, labels_query)
        test_accuracies.append(accs)

        # [b, update_step+1]
        accuracies = np.array(test_accuracies)
        accuracies = 100*accuracies[:,-1]
        avg = np.array(accuracies).mean().astype(np.float16)
        std = np.std(np.array(accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} %)' \
                  .format(i, args.epoch, avg, ci95))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--load_path', type=str, help='model save path',
                           default='/data/liuyong/TianPinzhuo/IJCAI/cmMAML_office31/best_model.pth.tar')

    args = argparser.parse_args()

    main()
