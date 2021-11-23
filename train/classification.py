import sys
from numpy import add

from torchvision.transforms.transforms import ToTensor
sys.path.insert(1,'/home.stud/kuntluka/DeepLearning/models')
sys.path.insert(1,'/home.stud/kuntluka/DeepLearning/utils')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader

from AlexNet import AlexNet
# from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, optimizer, criterion, train_loader, dev_loader, num_epochs, writer):
    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    data, labels = iter(train_loader).next()

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
        # for batch_idx, (data,labels) in loop:
        if True:
            data = data.to(device)
            labels = labels.to(device)
            # data = data.reshape(data.shape[0], -1)

            scores = model(data)
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predictions = scores.max(1)
            running_samples += predictions.size(0)
            running_correct += (predictions == labels).sum()
            running_loss += loss.detach()
            loop.set_description(f"Epoch {epoch} / {num_epochs}")

            # loop.set_postfix(loss=float(running_loss) / (batch_idx+1), acc= float(running_correct) / float(running_samples))

        print()
        add
        writer.add_scalar('training_loss', running_loss, epoch)
        print(f"Running loss {running_loss}")

        # writer.add_scalar('training_loss', float(running_loss) / len(train_loader), epoch)
        # val_accuracy = check_accuracy(dev_loader, model)
        # writer.add_scalar('validation_accuracy', val_accuracy, epoch)
        # writer.add_scalar('training_accuracy', float(running_correct) / float(running_samples), epoch)
        
        running_correct = 0
        running_loss = 0
        running_samples = 0

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        # print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()
    return float(num_correct) /float(num_samples)


if __name__ == '__main__':
    input_size = 784
    num_classes = 10
    learning_rate = 5e-4
    batch_size = 32
    num_epochs = 100

    cifar_val = datasets.CIFAR100("~/dataset/CIFAR-100", 'val', download=True)
    cifar_train = datasets.CIFAR100("~/dataset/CIFAR-100", 'train', download=True)

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        ToTensor()
    ])
    
    image_net_train = datasets.ImageNet("/datagrid/public_datasets/imagenet/imagenet_pytorch/", 'train', transform=train_transform)
    image_net_val = datasets.ImageNet("/datagrid/public_datasets/imagenet/imagenet_pytorch/", 'val', transform=val_transform)
    # img = image_net_val[0][0]
    train_loader = DataLoader(dataset=image_net_train, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=image_net_val, batch_size = batch_size*2, shuffle=False, num_workers=1)

    # t = transforms.ToTensor()
    # img_np = t(img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/MNIST/tensorboard_test')
    # writer.add_image('image net sample', img_np)

    print(torch.cuda.is_available())


    # train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # dev_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    # dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size*2, shuffle=False)

    # model = NN(input_size, num_classes)
    model = AlexNet(num_classes=1000, batch_norm=True)


    example_data, example_labels = iter(val_loader).next()
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('mnist_dataset', img_grid)
    writer.add_graph(model, example_data)
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, optimizer, criterion, train_loader, val_loader, num_epochs, writer)
    writer.close()




