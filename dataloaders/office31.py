from skimage import io as sio
import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import skimage.transform
from torch.utils.data import DataLoader

trainids = [6, 2, 0, 5, 4, 10, 14, 19, 12, 24, 16, 11, 28, 29, 9, 17, 15, 22, 20, 23]
valids = [3, 30, 25, 13, 7]
testids = [1, 8, 18, 21, 26, 27]



class Office31Dataset(Dataset):

    def __init__(self, num_classes, num_support, num_query, num_epoch, img_size=[84, 84], phase='train'):
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.num_epoch = num_epoch
        self.img_size = np.array(img_size)
        self.phase = phase

        data_root = ['/data/liuyong/TianPinzhuo/domain adaption/office31/amazon/images',
                     '/data/liuyong/TianPinzhuo/domain adaption/office31/dslr/images',
                     '/data/liuyong/TianPinzhuo/domain adaption/office31/webcam/images'
                     ]
        clsnames = os.listdir(data_root[0])
        self.datasetlen = len(data_root)

        if phase == 'train':
            ids = trainids

        elif phase == 'test':
            ids = testids

        elif phase == 'val':
            ids = valids

        self.all_supportfiles = []
        self.all_queryfiles = []

        for _ in range(self.num_epoch):
            classids = random.sample(ids, num_classes)
            trainroots = random.sample(data_root, 2)
            testroots = [ i for i in data_root if i not in trainroots ]

            clsids_pathes = [(clsid, os.path.join(trainroots[datasetid], clsnames[clsid])) for clsid in classids
                             for datasetid in range(len(trainroots))]


            train_pathes = []

            for clsroot in clsids_pathes:

                if len(os.listdir(clsroot[1])) >= self.num_support:
                    train_pathes.extend( [(classids.index(clsroot[0]), os.path.join(clsroot[1], name), ids.index(clsroot[0]))
                                                for name in random.sample(os.listdir(clsroot[1]), self.num_support)] )
                else:
                    train_pathes.extend([(classids.index(clsroot[0]), os.path.join(clsroot[1], name), ids.index(clsroot[0]))
                                               for name in random.choices(os.listdir(clsroot[1]),
                                                                         k = self.num_support)])


            labels_and_images = self.get_images(train_pathes, len(trainroots), self.num_support)
            random.shuffle(labels_and_images)

            self.all_supportfiles.append(labels_and_images)

            #-----------query-------------------------
            clsids_pathes = [(clsid, os.path.join(testroots[datasetid], clsnames[clsid])) for clsid in classids
                             for datasetid in range(len(testroots))]

            test_pathes = []
            for clsroot in clsids_pathes:

                if len(os.listdir(clsroot[1])) >= self.num_query:
                    test_pathes.extend( [(classids.index(clsroot[0]), os.path.join(clsroot[1], name), ids.index(clsroot[0]))
                                                for name in random.sample(os.listdir(clsroot[1]), self.num_query)] )
                else:
                    test_pathes.extend([(classids.index(clsroot[0]), os.path.join(clsroot[1], name), ids.index(clsroot[0]))
                                               for name in random.choices(os.listdir(clsroot[1]),
                                                                         k = self.num_query)])


            labels_and_images = self.get_images(test_pathes, len(testroots), self.num_query)
            random.shuffle(labels_and_images)

            self.all_queryfiles.append(labels_and_images)



    def get_images(self, filelist, len, sample_num):

        num = len * sample_num
        imgs = []
        for i in range(self.num_classes):
            tmp = filelist[i*num : (i+1)*num]
            random.shuffle(tmp)
            imgs.extend(random.sample(tmp, sample_num))
        return imgs

    def __len__(self):
        return self.num_epoch

    def Resize(self, image):

        image = skimage.transform.resize(image, self.img_size, order=1, preserve_range=True, mode='constant')

        return image

    def RandomRotate(self, image):
        angle = [0, 90, 180, 270]
        ro = random.randint(0, 3)
        image = skimage.transform.rotate(image, angle[ro], preserve_range=True)
        return image

    def RandomFilp(self, image, u=0.5):
        if random.random() < u:
            image = np.fliplr(image)

        return image

    def Normalize(self, image):
        image = image.astype(np.float32)
        image = image / 255
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return image

    def Loadimages(self, item, issupport):

        if issupport:
            labels = [li[0] for li in self.all_supportfiles[item]]
            filepathes = [li[1] for li in self.all_supportfiles[item]]
            clsids = [li[2] for li in self.all_supportfiles[item]]
            image_num = self.num_support *self.num_classes
        else:
            labels = [li[0] for li in self.all_queryfiles[item]]
            filepathes = [li[1] for li in self.all_queryfiles[item]]
            clsids = [li[2] for li in self.all_queryfiles[item]]
            image_num = self.num_query *self.num_classes

        images = []

        for i in range(image_num):
            image = sio.imread(filepathes[i], as_gray=False)

            if self.phase == 'train':
                image = self.Normalize(image)
                image = self.Resize(image)
                image = self.RandomRotate(image)
                image = self.RandomFilp(image)

            elif self.phase == 'test' or self.phase == 'val':
                image = self.Normalize(image)
                image = self.Resize(image)

            image = image.transpose((2, 0, 1))
            images.append(image)


        all_image_batch = np.array(images)
        all_clsid_batch = np.array(clsids)
        all_classlabel_batch = np.array(labels)

        return torch.from_numpy(all_image_batch), torch.from_numpy(all_clsid_batch) ,torch.from_numpy(all_classlabel_batch)

    def __getitem__(self, item):
        support, support_clsid, support_label = self.Loadimages(item, True)
        query, query_clsid, query_label = self.Loadimages(item, False)
        return support, support_clsid, support_label, query, query_clsid, query_label


if __name__ == '__main__':
    dDataset_train = Office31Dataset(num_classes=5, num_support=5, num_query=3,
                                   num_epoch=10, img_size=(84, 84), phase='train')
    dloader_train = DataLoader(dDataset_train, shuffle=True, num_workers=1, batch_size=1)

    for i, batch in enumerate(dloader_train):
        _, _, _= [x for x in batch]
        print("111")
