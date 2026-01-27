
from torchvision import transforms
import torch
from torchvision import datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torch.utils.data import Dataset
import os
from PIL import Image


class TinyImageNet(Dataset):
   def __init__(self, root, train=True, transform=None):
       self.train = train
       self.transform = transform
       self.data_dir = os.path.join(root, "train" if train else "val")
       self.images = []
       self.labels = []
       self.class_to_idx = dict()

       classpath = os.path.join(root, "wnids.txt")
       print(os.path.abspath(classpath))
       with open(classpath, "r") as f:
           minindex = 0
           for line in f.readlines():
               self.class_to_idx[line.strip()] = minindex
               minindex += 1

       if train:
           for class_dir in os.listdir(self.data_dir):
               class_path = os.path.join(self.data_dir, class_dir, "images")
               for img_name in os.listdir(class_path):
                   self.images.append(os.path.join(class_path, img_name))
                   self.labels.append(class_dir)
       else:
           val_annotations = os.path.join(self.data_dir, "val_annotations.txt")
           with open(val_annotations, "r") as f:
               for line in f.readlines():
                   img_name, label = line.split("\t")[:2]
                   self.images.append(os.path.join(self.data_dir, "images", img_name))
                   self.labels.append(label)
   def __len__(self):
       return len(self.images)
   def __getitem__(self, idx):
       img_path = self.images[idx]
       label = self.class_to_idx[self.labels[idx]]
       img = Image.open(img_path).convert("RGB")
       if self.transform:
           img = self.transform(img)
       return img, label




class data_loader(object):

    def __init__(self):
        self.exist_dataset = ['mnist', 'fashion_mnist', 'qmnist', 'cifar10', 'cifar100', 'svhn', 'tinyimagenet']
        self.path = {
            'tinyimagenet': 'tinyimagenet'
        }

    def __call__(self, data_name=None, data_dir=None, download=True, size=None, train_batch=None, test_batch=None, normalize=False):

        self.normalize = normalize

        if data_name is None and data_dir is None:
            raise ValueError

        if data_name not in self.exist_dataset:
            raise ValueError

        if data_name:
            train_loader, test_loader = self.__load(data_name, download, size, train_batch, test_batch)

        if data_dir:
            self.data_dir = data_dir

        if self.normalize:
            self.type = f"{data_name} with normalization"
        else:
            self.type = f"{data_name} without normalization"

        return train_loader, test_loader

    def __load(self, name, download, size, train_batch, test_batch):

        if train_batch is None:
            train_batch = 128

        if test_batch is None:
            test_batch = 200

        self.train_batch = train_batch
        self.test_batch = test_batch

        if name == 'mnist':
            if size is None:
                size = 28
            train, test = self.__load_mnist(download, size, train_batch, test_batch)
        elif name == 'fashion_mnist':
            if size is None:
                size = 28
            train, test = self.__load_fashion_mnist(download, size, train_batch, test_batch)
        elif name == 'qmnist':
            if size is None:
                size = 28
            train, test = self.__load_qmnist(download, size, train_batch, test_batch)
        elif name == 'cifar10':
            if size is None:
                size = 32
            train, test = self.__load_cifar10(download, size, train_batch, test_batch)
        elif name == 'cifar100':
            if size is None:
                size = 32
            train, test = self.__load_cifar100(download, size, train_batch, test_batch)
        elif name == 'svhn':
            if size is None:
                size = 32
            train, test = self.__load_svhn(download, size, train_batch, test_batch)
        elif name == 'tinyimagenet':
            if size is None:
                size = 64
            train, test = self.__load_tinyimagenet(download, size, train_batch, test_batch)

        return train, test

    # function for loading mnist dataset
    def __load_mnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                               transforms.Resize((size, size)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                               transforms.Resize((size, size)),
                               transforms.ToTensor(),
                           ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.path['mnist_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.path['mnist_path'], train=False, download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    # function for loading fashion mnist dataset
    def __load_fashion_mnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                      transforms.Resize((size, size)),
                                      transforms.RandomHorizontalFlip(),       
                                      transforms.RandomRotation(15),  
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                  ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                                      transforms.Resize((size, size)),
                                      transforms.RandomHorizontalFlip(),       
                                      transforms.RandomRotation(15), 
                                      transforms.ToTensor(),
                                  ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(self.path['fashion_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(self.path['fashion_path'], train=False, download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    # function for loading qmnist dataset
    def __load_qmnist(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.QMNIST(self.path['mnist_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.QMNIST(self.path['mnist_path'], train=False, download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )
        return train_loader, test_loader

    # function for loading cifar10 dataset
    def __load_cifar10(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])
        else:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.path['cifar10_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.path['cifar10_path'], train=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader

    # function for loading cifar100 dataset
    def __load_cifar100(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])
        else:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(self.path['cifar100_path'], train=True, download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(self.path['cifar100_path'], train=False, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader

    # function for loading svhn dataset
    def __load_svhn(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 # transforms.RandomCrop(size, padding=4),
                                 # transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])
        else:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 # transforms.RandomCrop(size, padding=4),
                                 # transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                             ])

        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(self.path['svhn_path'], split='train', download=download, transform=train_transform),
            batch_size=train_batch_size, shuffle=True
        )

        # extra_loader = torch.utils.data.DataLoader(
        #     datasets.SVHN(self.path['svhn_path'], split='extra', download=download, transform=train_transform),
        #     batch_size=train_batch_size, shuffle=True
        # )

        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(self.path['svhn_path'], split='test', download=download, transform=test_transform),
            batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader


    # function for loading tinyimagenet dataset
    def __load_tinyimagenet(self, download, size, train_batch_size, test_batch_size):

        if self.normalize:
            train_transform = transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.RandomRotation(15),
                                 transforms.RandomCrop(size, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])

            test_transform = transforms.Compose([
                                 # ToTensor: Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                             ])
        else:
            train_transform = transforms.Compose([
                                transforms.RandomCrop(size, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])
            test_transform = transforms.Compose([
                                transforms.ToTensor()])
        train_dataset = TinyImageNet(root=self.path['tinyimagenet'], train=True, transform=train_transform)
        test_dataset = TinyImageNet(root=self.path['tinyimagenet'], train=False, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        # train_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(root=self.path['tinyimagenet'] + 'train', transform=train_transform),
        #     batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(root=self.path['tinyimagenet'] + 'val', transform=test_transform),
        #     batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader


if __name__ == '__main__':

    data_selection = data_loader()
    # train_loader, test_loader = data_selection('svhn')
    names = ['mnist', 'fashion_mnist', 'qmnist', 'cifar10', 'cifar100', 'svhn']
    for name in names:
        train_loader, test_loader = data_selection(name)

    data, target = next(iter(train_loader))
    print(data)
    print(target)

