import os
from torchvision.datasets import CIFAR10, STL10
import numpy as np


class STL10PAIR(STL10):
    """SRL10 Dataset.
    """

    def __init__(self, root, split='labeled', folds=None, transform=None, target_transform=None, download=True):
        self.root = root
        self.split = split
        self.folds = folds
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        # now load the picked numpy arrays
        if self.split == 'labeled':
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            test_data, test_labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])
            self.data = np.concatenate((self.data, test_data))
            self.labels = np.concatenate((self.labels, test_labels))
        else:  # self.split == 'test':
            self.test_data, self.test_labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def __load_folds(self, folds):
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.uint8, sep=' ')
            self.data, self.labels = self.data[list_idx, :, :, :], self.labels[list_idx]

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
            # class_name = self.classes[target]
        else:
            img, target = self.data[index], 255 # 255 is an ignore index
            # class_name = 'unlabeled'


        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index