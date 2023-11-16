import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# dataset path
data_dir = './data/exquisite125'

# use float32 to simulate fake int8 datetype
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [1./255., 1./255., 1./255.])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [1./255., 1./255., 1./255.])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x],
                             batch_size=16,
                             shuffle=True if x == 'train' else False,
                             num_workers=6,
                             pin_memory=True)
              for x in ['train', 'val']}



if __name__ == '__main__':
    loader = dataloaders['val']
    for imgs, labels in loader:
        print(labels)
        break
    