import torch 
import torchvision.datasets as data
import torchvision.transforms as tranforms
import torchvision.utils
from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


def get_dataset_mean_std(dataloader : DataLoader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std

def log_sample_images(dataset, writer : SummaryWriter, name : str = 'Dataset sample', num_samples : int = 12, resolution = (224,224)):
    img_samples = [dataset[i][0] for i in range(num_samples)]
    # img_labels = [dataset[i][1] for i in range(num_samples)]

    img_transforms = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor()
    ])
    img_samples_transformed = list(map(img_transforms, img_samples))
    grid = torchvision.utils.make_grid(img_samples_transformed)
    writer.add_image(name, grid)


