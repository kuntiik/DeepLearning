from pytorch_lightning import trainer
from pytorch_lightning import callbacks
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch._C import TracingState
from torch.utils.data.dataset import random_split
from torchmetrics.classification import accuracy
from torchvision.datasets.cifar import CIFAR100
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ToTensor

from train.train import HyperParameters, NetworkTraining
from models.AlexNet import AlexNet
import utils.utils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
import torchmetrics


class NeuralNet(pl.LightningModule):
    def __init__(self, num_classes, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.ce_loss = nn.CrossEntropyLoss()
        self.model = AlexNet(num_classes, 0.3, True)
    
    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimiser

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.ce_loss(logits, y)
        # return pl.TrainResult(loss)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.ce_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = torchmetrics.Accuracy().to(self.device)
        acc = accuracy(preds, y)
        # result = pl.EvalResult(checkpoint_on = loss)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 100
        self.image_dims = (224, 224)

        self.cifar100_mean = torch.tensor([0.5071, 0.4865, 0.4409])
        self.cifar100_std = torch.tensor([0.2673, 0.2564, 0.2761])
        self.transform = transforms.Compose([
            transforms.Resize((self.image_dims)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
        ])
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            CIFAR100_full = datasets.CIFAR100(self.data_dir, train=True, transform=self.transform)
            self.CIFAR100_train, self.CIFAR100_val = random_split(CIFAR100_full, [len(CIFAR100_full) - 5000, 5000])
        
        if stage == 'test' or stage is not None:
            self.CIFAR100_test = datasets.CIFAR100(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.CIFAR100_train, batch_size=self.batch_size, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.CIFAR100_val, batch_size=self.batch_size, num_workers=32)
    
    def test_loader(self):
        return DataLoader(self.CIFAR100_test, batch_size=self.batch_size, num_workers=32)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 1000
        self.image_dims = (224, 224)
        self.image_net_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_net_std = torch.tensor([0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((self.image_dims)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_net_mean, std=self.image_net_std)
        ])
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            ImageNet_full = datasets.ImageNet(self.data_dir, 'train', transform=self.transform)
            self.ImageNet_train, self.ImageNet_val = random_split(ImageNet_full, [len(ImageNet_full) - 5000, 5000])
        
        if stage == 'test' or stage is not None:
            self.ImageNet_test = datasets.ImageNet(self.data_dir, 'val', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.ImageNet_train, batch_size=self.batch_size, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.ImageNet_val, batch_size=self.batch_size, num_workers=32)
    
    def test_loader(self):
        return DataLoader(self.ImageNet_test, batch_size=self.batch_size, num_workers=32)

if __name__ == '__main__':
    dm = CIFAR100DataModule('~/dataset/CIFAR-100', batch_size=64)
    # dm_image_net = ImageNetDataModule("/datagrid/public_datasets/imagenet/imagenet_pytorch/")
    model = NeuralNet(100, 0.1)
    trainer = pl.Trainer(max_epochs = 40, gpus=1, auto_select_gpus=True, auto_lr_find=True, benchmark=True, callbacks=[ModelSummary(max_depth=-1)])
    trainer.tune(model, dm)
    trainer.fit(model, dm)