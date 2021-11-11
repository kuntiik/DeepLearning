import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
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

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
        for batch_idx, (data,labels) in loop:
            data = data.to(device)
            labels = labels.to(device)
            data = data.reshape(data.shape[0], -1)

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

            loop.set_postfix(loss=float(running_loss) / (batch_idx+1), acc= float(running_correct) / float(running_samples))

        writer.add_scalar('training_loss', float(running_loss) / len(train_loader), epoch)
        val_accuracy = check_accuracy(dev_loader, model)
        writer.add_scalar('validation_accuracy', val_accuracy, epoch)
        writer.add_scalar('training_accuracy', float(running_correct) / float(running_samples), epoch)
        
        running_correct = 0
        running_loss = 0
        running_samples = 0

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training dataset")
    else:
        print("Checking accuracy on development dataset")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        # print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        return float(num_correct) /float(num_samples)
    model.train()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/MNIST/tensorboard_test')

    input_size = 784
    num_classes = 10
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 5

    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size*2, shuffle=False)

    model = NN(input_size, num_classes).to(device)

    example_data, example_labels = iter(dev_loader).next()
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('mnist_dataset', img_grid)
    writer.add_graph(model, example_data.reshape(example_data.shape[0],-1))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, optimizer, criterion, train_loader, dev_loader, num_epochs, writer)
    writer.close()




