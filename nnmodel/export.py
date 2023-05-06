import os.path

import torch
from torchvision import datasets
from torchvision import transforms

from .network import NNClassifier

pretrained = 'nnmodel/pretrained.pth'

def _load_model(net: NNClassifier, pretrained: str, device):
    assert os.path.exists(pretrained), 'The pretrained path can not be empty'.format(pretrained)
    net.load_state_dict(torch.load(pretrained))
    return net.to(device)


def load_model(device):
    return _load_model(NNClassifier(), pretrained, device)


def load_data(is_download=False):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),

        "test": transforms.Compose([transforms.ToTensor()])}
    project_root = os.path.abspath(os.getcwd())  # get data root path

    data_root = os.path.join(project_root, 'nnmodel')
    train_dataset = datasets.FashionMNIST(root=data_root, train=True, transform=data_transform['train'], download=is_download)
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, transform=data_transform['test'])

    return train_dataset, test_dataset

if __name__ == '__main__':
    net = load_model('cuda:0')

