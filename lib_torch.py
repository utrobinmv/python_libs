import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

import tqdm 

def creat_models(features, classes):
    model = nn.Sequential(
        nn.Linear(features, 64),
        nn.ReLU(),
      nn.Linear(64, classes)
    )
    return model

def torch_vector_to_numpy(vector):
    return vector.numpy()

def numpy_array_to_torch_vector(data):
    return torch.from_numpy(data)

def create_dataloader(train):
    '''
    Создает объект дата лоадер

    Другие примеры
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    '''
    batch_size = 128
    train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
    x_batch, y_batch = next(iter(train_loader))
    print(x_batch.shape, y_batch.shape)
    return train_loader


def torch_shed(num, optimizer_ft):
    '''
    Возвращает шедуллер скорости обучения

    Пример использования
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
    torch_shed(1,optimizer_ft)
    '''



    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def torch_optim(num, model):
    '''
    Функция возвращает оптимизатор Торча
    '''
    if num == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    elif num == 2:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    elif num == 3:
        optimizer = torch.optim.Adagrad()
    elif num == 4:
        optimizer = torch.optim.RMSprop()

    return optimizer

def torch_loss(num):

    loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.BCELoss()
    return loss_function

def torch_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion


def model_summary(model, features):
    '''
    Показывает структуру нейронной сети
    features - число, количество признаков, передается так как features - является динамическим параметром сети

    '''
    summary(model, (features,), batch_size=228)


def model_save(model, save_name):
    '''
    Сохраняет модель
    torch.save(model.state_dict(), 'AlexNet_fine_tune.pth')
    '''

    torch.save(model.state_dict(), save_name)


def model_load(model, save_name):
    '''
    Загружает модель
    model.load_state_dict(torch.load('AlexNet_fine_tune.pth'))
    '''

    model.load_state_dict(torch.load(save_name))



def model_transfer_learning(model_extractor, num_features):
    '''
    Пример
    model_extractor = models.alexnet(pretrained=True)


    Просмотр все ли параметры имеют способность к обучению
    '''

    for param in model_extractor.parameters():
        print(param.requires_grad)   

    # замораживаем параметры (веса)
    for param in model_extractor.parameters():
        param.requires_grad = False      

    num_features = 9216
    # Заменяем Fully-Connected слой на наш линейный классификатор
    model_extractor.classifier = torch.nn.Linear(num_features, 2) 

def model_transfer_learning_layers_to_unfreeze(model_mixed, num_features):
    '''
    Замораживает часть слоев сети
    '''
    layers_to_unfreeze = 5

    # Выключаем подсчет градиентов для слоев, которые не будем обучать
    for param in model_mixed.features[:-layers_to_unfreeze].parameters():
        param.requires_grad = False

    # num_features -- это размерность вектора фич, поступающего на вход FC-слою
    num_features = 9216
    # Заменяем Fully-Connected слой на наш линейный классификатор
    model_mixed.classifier = nn.Linear(num_features, 2)    



def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    '''
    Обучение модели
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

    info = fit(10, model, criterion, optimizer, *get_dataloaders(4))
    plot_trainig(*info)
    '''
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    train_losses = []
    val_losses = []
    valid_accuracies = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for xb, yb in tqdm(train_dl):
            xb, yb = xb.to(device), yb.to(device)

            loss = loss_func(model(xb), yb)
            loss_sum += loss.item()

            loss.backward()
            opt.step()
            opt.zero_grad()
        train_losses.append(loss_sum / len(train_dl))

        model.eval()
        loss_sum = 0
        correct = 0
        num = 0
        with torch.no_grad():
            for xb, yb in tqdm(valid_dl):
                xb, yb = xb.to(device), yb.to(device)

                probs = model(xb)
                loss_sum += loss_func(probs, yb).item()

                _, preds = torch.max(probs, axis=-1)
                correct += (preds == yb).sum().item()
                num += len(xb)

        val_losses.append(loss_sum / len(valid_dl))
        valid_accuracies.append(correct / num)

    return train_losses, val_losses, valid_accuracies

def fit__with_shed(epochs, model, loss_func, opt, train_dl, valid_dl, lr_sched=None):
    '''
    Обучение с изменяемой скоростью обучения
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=0.5)

    info = fit(10, model, criterion, optimizer, *get_dataloaders(4))
    plot_trainig(*info)
    '''
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    train_losses = []
    val_losses = []
    valid_accuracies = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for xb, yb in tqdm(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_func(model(xb), yb)
            loss_sum += loss.item()

            loss.backward()
            opt.step()
            opt.zero_grad()
        train_losses.append(loss_sum / len(train_dl))

        model.eval()
        loss_sum = 0
        correct = 0
        num = 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                probs = model(xb)
                loss_sum += loss_func(probs, yb).item()

                _, preds = torch.max(probs, axis=-1)
                correct += (preds == yb).sum().item()
                num += len(xb)

        val_losses.append(loss_sum / len(valid_dl))
        valid_accuracies.append(correct / num)

        # CHANGES HERE
        lr_ched.step()
        # CHANGES END

    return train_losses, val_losses, valid_accuracies

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):

    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    #Ваш код здесь
    losses = {'train': [], "val": []}

    pbar = tqdm.autonotebook.trange(num_epochs, desc="Epoch:")

    for epoch in pbar:

        # каждя эпоха имеет обучающую и тестовую стадии
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # установаить модель в режим обучения
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # итерируемся по батчам
            for data in tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter:"):
                # получаем картинки и метки
                inputs, labels = data

                # оборачиваем в переменные
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = inputs, labels

                # инициализируем градиенты параметров
                if phase=="train":
                    optimizer.zero_grad()

                # forward pass
                if phase == "eval":
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                preds = torch.argmax(outputs, -1)
                loss = criterion(outputs, labels)

                # backward pass + оптимизируем только если это стадия обучения
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # статистика
                running_loss += loss.item()
                running_corrects += int(torch.sum(preds == labels.data))

            epoch_loss = running_loss / len(dataloaders) #Деление на количество объектов в датасете
            epoch_acc = running_corrects / len(dataloaders) #Деление на количество объектов в датасете

            # Ваш код здесь
            losses[phase].append(epoch_loss)

            pbar.set_description('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            # если достиглось лучшее качество, то запомним веса модели
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))

    # загрузим лучшие веса модели
    model.load_state_dict(best_model_wts)
    return model, losses

#Модули для работы с большинством моделей PyTorch

class LoadFilesDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    Ососбенность в том, что метки представлены в виде строк
    Пример использования:
    
    from pathlib import Path
    TRAIN_DIR = Path('train/dataset')
    files_list = {'train_val': sorted(list(TRAIN_DIR.rglob('*.jpg'))),
              'test': sorted(list(TEST_DIR.rglob('*.jpg')))
              }
    labels_list = {'train_val': [path.parent.name for path in files_list['train_val']]}x
    files_list['train'], files_list['val'] = train_test_split(files_list['train_val'], test_size=0.25, \
                                          stratify=labels_list['train_val'])
    datasets_list = {'train': LoadFilesDataset(files_list['train'], mode='train'), train_label_encoder = LabelEncoder()}

    dataloaders_wht = {'train': DataLoader(datasets_list['train'], batch_size=BATCH_SIZE, sampler=MySamplerTrain),
               'val': DataLoader(datasets_list['val'], batch_size=BATCH_SIZE, shuffle=False)}

    """
    def __init__(self, files, mode, train_label_encoder):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
     
        #self.label_encoder = LabelEncoder()
        
        #if self.mode != 'train':
        self.label_encoder = train_label_encoder

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            
        if self.mode == 'train':
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)
                      
    def __len__(self):
        return self.len_
      
    def _load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
  
    def get_label(self, index):
        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        return y

    def __getitem__(self, index):
        #print('index:', index)
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_LIST, IMAGE_STD_LIST) 
        ])
            
        x = self._load_sample(self.files[index])
        if self.mode == 'train':
            album = transforms.Compose([
                #transforms.RandomResizedCrop(500),
                transforms.RandomHorizontalFlip(),
            ])
            #print(type(x))
            x = album(x)
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)
    
    
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    '''
    Sampler позволяет сбалансировать частично классы в выборке DataLoader
    Обязательно требуется аугументация изображений в датасете
    Пример использования:
    def callback_get_label(dataset, idx):
       #label = dataset.labels[idx]
       label = dataset.get_label(idx)
       return label
    MySamplerTrain=ImbalancedDatasetSampler(dataset=datasets_list['train'], callback_get_label=callback_get_label)
    dataloader = DataLoader(datasets_list['train'], batch_size=BATCH_SIZE, sampler=MySamplerTrain)
    '''
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
        
        Данные sampler позволяет строить через DataLoader более сбалансированную выборку классов, хотя не идеальную!
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
        self.num = np.zeros(self.weights.shape)
        
        #print(label_to_count)
        #print(self.num_samples)
        
        #print(self.weights.shape)
        #print(len(self.indices))

    def _get_label(self, dataset, idx):
        #print('get_label',idx)
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError

    def _return_indices(self, i):
        #print(int(i))
        #self.num[int(i)] += 1 #анализ показал, что выбор конкретного файла происходит случайно с учетом весов
        #Но некая балансовость по классам наблюдается, хотя по файлам балансовости нет.
        #Есть куда улучшать
        return self.indices[i]
            
    def __iter__(self):
        #print('iter')
        #res = (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
        res = (self._return_indices(i) for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
        return res

    def __len__(self):
        return self.num_samples