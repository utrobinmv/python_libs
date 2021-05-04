import torchvision.transforms as tfs
import torchvision
from torchvision.datasets import MNIST

import os

data_tfs = tfs.Compose([
  tfs.ToTensor(),
  tfs.Normalize((0.5), (0.5))
])

transform = tfs.Compose(
    [tfs.ToTensor()])

data_transforms = {
    'train': tfs.Compose([
        tfs.RandomResizedCrop(244),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': tfs.Compose([
        tfs.Resize(256),
        tfs.CenterCrop(244),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_transforms2 = {
    'train': tfs.Compose([
        tfs.RandomResizedCrop(299),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': tfs.Compose([
        tfs.Resize(299),
        tfs.CenterCrop(299),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_dataset_mnist(root):
   train = MNIST(root, train=True,  transform=data_tfs, download=True)
   test  = MNIST(root, train=False, transform=data_tfs, download=True)
   return train, test



def load_dataset_cifar10():
   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
   return trainset


def load_models(num):
   if num == 1:
      model = torchvision.models.alexnet(pretrained=True)
   elif num == 2:      
      model = torchvision.models.vgg16(pretrained=True)
   elif num == 3:
      model = torchvision.models.inception_v3(pretrained=True)
      
      
   return model

def load_from_folder(data_dir):
   '''
   Пример использования результата
   # специальный класс для загрузки данных в виде батчей
   dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
               shuffle=True, num_workers=2)
               for x in ['train', 'val']}
   '''
   image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}   
   return image_datasets