import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.profiler import profiler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.profiler

class HyperParameters:
    def __init__(self, num_epochs : int, learning_rate : float, batch_size : int):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

class NetworkTraining:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader,
                writer : SummaryWriter, hparams : HyperParameters, note : str = "", checkpoint = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.best_loss = np.inf
        self.checkpoint_interval = 3
        self.start_saving_at = 15
        self.note= note

        if checkpoint != None:
            self.load_checkpoint(self, checkpoint)



    def single_batch_overfitting(self):
        data, targets = iter(self.train_loader).next()
        data = data.to(self.device)
        targets = targets.to(self.device)
        data_len = len(data)
        targets = targets.to(self.device)
        loop = tqdm(range(self.hparams.num_epochs), total=self.hparams.num_epochs)
        for epoch in loop:
            scores = self.model(data)
            loss = self.criterion(scores, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, predictions = scores.max(1)
            num_correct = (predictions == targets).sum()
            accuracy = num_correct / data_len

            self.writer.add_scalar(f"""Single batch training loss: LR: 
            {self.hparams.learning_rate} batch_size : {self.hparams.batch_size}""", loss.item(), epoch)
            self.writer.add_scalar(f"""Single batch training accuracy: LR: 
            {self.hparams.learning_rate} batch_size : {self.hparams.batch_size}""", accuracy, epoch)
            
            loop.set_postfix(loss=loss.item(), acc=float(accuracy))
    
    def save_checkpoint(self, type : str):
        checkpoint = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        foldername = "/home.stud/kuntluka/DeepLearning/Checkpoints" +  self.model.name + self.train_loader.name + self.note
        if not(os.path.isdir(foldername)):
            os.mkdir(foldername)
        checkpoint_name = os.path.join(foldername, type + ".pth")
        torch.save(checkpoint, checkpoint_name)

    def load_checkpoint(self, checkpoint):
        checkpoint_state_dict = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_state_dict["model"])
        self.optimizer.load_state_dict(checkpoint_state_dict["optimizer"])


        
    def training(self):
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/imagenet'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
            # activities=[
            #     torch.profiler.ProfilerActivity.CPU,
            #     torch.profiler.ProfilerActivity.CUDA,
            # ],
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./results'),
            # record_shapes = True
        ) as p:

            for epoch in range(self.hparams.num_epochs):
                loop = tqdm(enumerate(self.train_loader), leave=False, total=len(self.train_loader))
                loop.set_description(f"Epoch {epoch} / {self.hparams.num_epochs}")

                for batch_idx, (data, targets) in loop:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    scores = self.model(data)
                    loss = self.criterion(scores, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    _, predictions = scores.max(1)
                    running_samples += predictions.size(0)
                    running_correct += (predictions == targets).sum()
                    running_loss += loss.item()

                    # if batch_idx == 80:
                    #     sys.exit()

                    if batch_idx % 100 == 99:
                        current_step = batch_idx+epoch*len(self.train_loader)
                        self.writer.add_scalar('Running training loss', running_loss / (batch_idx+1), current_step)
                        self.writer.add_scalar('Runing training accuracy', float(running_correct) / running_samples, current_step)

                    loop.set_postfix(loss=running_loss/(batch_idx+1), acc=float(running_correct)/float(running_samples))

                    p.step()
                
                self.writer.add_scalar(f"""Training loss: LR: {self.hparams.learning_rate} batch_size : 
                    {self.hparams.batch_size}""", running_loss / len(self.train_loader), epoch)
                self.writer.add_scalar(f"""Training accuracy: LR: {self.hparams.learning_rate} batch_size : 
                    {self.hparams.batch_size}""", float(running_correct) / float(running_samples), epoch)
                self.check_val_accuracy_loss(epoch)
                running_correct = 0
                running_loss = 0
                running_samples = 0

                if epoch % self.checkpoint_interval == self.checkpoint_interval - 1:
                    self.save_checkpoint('regular')
                

    def check_val_accuracy_loss(self, epoch):
        num_correct = 0
        running_loss = 0
        self.model.eval()

        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, total=len(self.val_loader), leave=False):
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(data)
                loss = self.criterion(scores, targets)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                running_loss += loss.item()
                # num_samples += predictions.size(0)
        self.model.train()
        validation_loss = running_loss / len(self.val_loader)
        self.writer.add_scalar(f"Validation loss: LR: {self.hparams.learning_rate} batch_size : \
            {self.hparams.batch_size}", validation_loss, epoch)
        self.writer.add_scalar(f"Validation accuracy: LR: {self.hparams.learning_rate} batch_size : \
            {self.hparams.batch_size}", num_correct / len(self.val_loader.dataset), epoch)
        
        if epoch >= self.start_saving_at and validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.save_checkpoint('best')

            

