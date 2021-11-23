import torch
import torch.nn as nn
import torch.optim as optim
from torch._C import TracingState
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ToTensor

from train.train import HyperParameters, NetworkTraining
from models.AlexNet import AlexNet
import utils.utils
# import torch.profiler



writer = SummaryWriter('runs/CIFAR-100/AlexNet')
hyper_parameters = HyperParameters(100, 0.001, 32)

cifar_path = "~/dataset/CIFAR-100"
cifar100_mean = torch.tensor([0.5071, 0.4865, 0.4409])
cifar100_std = torch.tensor([0.2673, 0.2564, 0.2761])
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
])

validation_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
])
dataset_train = datasets.CIFAR100(cifar_path, 'train', transform=train_transforms)
dataset_val = datasets.CIFAR100(cifar_path, 'val', transform=validation_transforms)
# dataset_train = datasets.ImageNet("/datagrid/public_datasets/imagenet/imagenet_pytorch/", 'train', transform=train_transforms)
# dataset_val = datasets.ImageNet("/datagrid/public_datasets/imagenet/imagenet_pytorch/", 'val', transform=None)

train_loader = DataLoader(dataset=dataset_train, batch_size=hyper_parameters.batch_size,
    shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=dataset_val, batch_size=hyper_parameters.batch_size*2,
    shuffle=False, num_workers=1)
# utils.utils.log_sample_images(dataset_val, writer, 'ImageNet sample')
# mean, std = utils.utils.get_dataset_mean_std(train_loader)

train_loader.name = "CIFAR-100"
model = AlexNet(100, 0.3, True)
network_train = NetworkTraining(
        model,
        optim.Adam(model.parameters(), lr=hyper_parameters.learning_rate),
        nn.CrossEntropyLoss(),
        train_loader,
        val_loader,
        writer,
        hyper_parameters,
        note="BasicAlexNet"
    )
# network_train.single_batch_overfitting()
# network_train.check_val_accuracy_loss()
network_train.save_checkpoint('test')
network_train.training()

print('done')
