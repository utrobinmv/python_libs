import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from matplotlib.colors import ListedColormap

from sklearn import metrics

def plot_scater(x_coords, y_coords):
    '''
    Рисует точки на плоскости, с заданными координатами
    '''
    plt.scatter(x_coords, y_coords, label='train', color = 'r')
    
    
def plot_plot(x_coords, y_coords):
    '''
    Рисует линии из точек, с заданными координатами
    каждая следующая точка соединяется с предыдущей
    '''
    plt.plot(x_coords, y_coords, label='real', color = 'r')
    
    
def plot_instruction():
    plt.figure(figsize=(20, 7)) # Первое число пропорция полотна по горизонтали, вторая пропорция по вертикали.

    #xx, yy = np.meshgrid(np.linspace(np.min(X_train[names[0]]) - eps, np.max(X_train[names[0]]) + eps, 500),
    #                 np.linspace(np.min(X_train[names[1]]) - eps, np.max(X_train[names[1]]) + eps, 500))
    #xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]

    ax = None

    for i, types in enumerate([['train', 'test'], ['train'], ['test']]):
        ax = plt.subplot(1, 3, i + 1, sharey=ax) # Разделяет зону на три графика по горизонтали и 1 по вертикали. Третье число, это номер графика
        if 'train' in types:
            plt.scatter(X_train, y_train, label='train', c='b') # Рисует точки
        if 'test' in types:
            plt.scatter(X_test, y_test, label='test', c='orange') # Рисует точки

        plt.plot(X, linear_expression(X), label='real', c='g') # Рисует линию
        plt.plot(X, regressor.predict(X[:, np.newaxis]), label='predicted', c='r') # Рисует линию
        
        plt.pcolormesh(xx, yy, Z, cmap=plt.get_cmap('viridis')) #Добавляет некий градиент цветов
        plt.colorbar() # Добавляет colorbar с используемыми градиентами
        
        plt.ylabel('target') #Справа от графика расшифровывает обозначение оси абсцисс Y
        plt.xlabel('feature') #Снизу от графика расшифровывает обозначение оси абсцисс X
        plt.title(" ".join(types)) #Сверху отображает заголовок графика
        plt.grid(alpha=0.2) # Показывает будет ли сетка на графике
        plt.ylim((0, 150)) #Задает лимит значений по отображаемым осям координат
        plt.legend() # Отображает легенду на самом графике каким цветом какие линии обозначены

    plt.show()

def plot_2d_dataset(list_train, list_label):
    '''
    Рисует точки из списка list_train' (x,y)
    в виде графика точек 8 на 8 см размерность
    и рисует разных цветов, в зависимости от 0 и 1 данных label
    '''
    colors_label = matplotlib.colors.ListedColormap(['red', 'blue', 'green', 'yellow'])
    plt.figure(figsize=(8,8)) #Размеры
    plt.scatter(list(map(lambda x:x[0], list_train)), list(map(lambda x:x[1], list_train)), c=list_label, cmap=colors_label)


def plot_visible_classification(x_coords, y_coords, class_labels):
    '''
    Рисует облако точек по x и y координатам
    Где метки классов переданы в переменной class_labels
    '''
    colors = matplotlib.colors.ListedColormap(['red', 'blue', 'green'])
    plt.figure(figsize=(8,6))
    plt.scatter(x_coords, y_coords, c = class_labels, cmap = colors, s=50)
    
    
def roc_auc_print(y_train, y_train_predicted, y_test, y_test_predicted, roc_auc_score):
    '''
    Выводит график обучение по roc_auc_score
    Пример использования:
        roc_auc_print(y_train, y_train_predicted, y_test, y_test_predicted, metrics.roc_auc_score)
    '''
    
    train_auc = roc_auc_score(y_train, y_train_predicted)
    test_auc = roc_auc_score(y_test, y_test_predicted)
    plt.figure(figsize=(10,7))
    plt.plot(*roc_curve(y_train, y_train_predicted)[:2], label='train AUC={:.4f}'.format(train_auc))
    plt.plot(*roc_curve(y_test, y_test_predicted)[:2], label='test AUC={:.4f}'.format(test_auc))
    legend_box = plt.legend(fontsize='large', framealpha=1).get_frame()
    legend_box.set_facecolor("white")
    legend_box.set_edgecolor("black")
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))
    plt.show()

def visible_matrix_hist(df, list_futures, list_target):
    plt.figure(figsize=(20,24)) #Размеры
    
    plot_number = 0
    for feature_name in list_futures:
        for target_name in list_target:
            plot_number += 1
            plt.subplot(4,3, plot_number) #Выводит график в матрицу 4(строки)х3(колонки) в конкретное место от 1..12
            plt.hist(df[df.target_name == target_name][feature_name]) #Фильтр по pandas данных по конкретному значению колонки и конкретному признаку
            plt.title(target_name)
            plt.xlabel('cm')
            plt.ylabel(feature_name[:-4]) # feature_name - это строка, выводит значение без последних 4х символов
            
            
def visible_hist_categorial(data, cat_cols):
    '''
    # Построим гистограммы для категориальных признаков
    '''
    f,ax = plt.subplots(8, 2, figsize=(20,40))
    for ax, col in zip(ax.ravel(), cat_cols):
        dat = data[col].value_counts()
        sns.barplot(y=dat.index, x=dat.values, alpha=0.6, ax=ax)
        ax.set_title(col, fontsize=16)
        ax.tick_params(labelsize=14)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)            

def scatter(actual, predicted, T):
    '''
    Рисует один scatter plot
    plt.figure(figsize=(5, 5))
    scatter(actual_0, predicted_0, 0.5)
    '''
    plt.scatter(actual, predicted)
    plt.xlabel("Labels")
    plt.ylabel("Predicted probabilities")
    plt.plot([-0.2, 1.2], [T, T])
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    
def many_scatters(actuals, predicteds, Ts, titles, shape):
    '''
    Рисует несколько scatter plot в таблице, имеющей размеры shape
    many_scatters([actual_0, actual_1, actual_2], [predicted_0, predicted_1, predicted_2], 
              [0.5, 0.5, 0.5], ["Perfect", "Typical", "Awful algorithm"], (1, 3))
    '''
    plt.figure(figsize=(shape[1]*5, shape[0]*5))
    i = 1
    for actual, predicted, T, title in zip(actuals, predicteds, Ts, titles):
        ax = plt.subplot(shape[0], shape[1], i)
        ax.set_title(title)
        i += 1
        scatter(actual, predicted, T)
        
def visible_train_test_labels(train_labels, test_labels):
    '''
    Выводит два простых графика гистрограм, рядом, с отображением меток классов train_labels и test_labels
    '''
    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    plt.hist(train_labels)
    plt.title('train data')
    
    plt.subplot(1,2,2)
    plt.hist(test_labels)
    plt.title('test data')
    

def visible_regression_line(train_labels, test_labels, predict_train_data, predict_test_data):
    '''
    Выводит результат обученной модели, в виде облака точек регресси
    если данное облако похоже на некую линию, то это может сигнализировать, что модель неплохо предсказывает.
    чем лучше данные похожи на линию, тем лучше предсказывает модель
    '''
    plt.figure(figsize=(10,6))
    # plt.subplot(1,1,1)
    plt.grid(True)
    plt.scatter(train_labels, predict_train_data, alpha=0.5, color = 'red')
    plt.scatter(test_labels, predict_test_data, alpha=0.5, color = 'blue')
    plt.title('no parameters setting')
    plt.xlim(-100,1100)
    plt.ylim(-100,1100)
    
    
    
#Решающие деревья
def get_meshgrid(data, step=.05, border=.5,):
    '''
    Вспомогательная функция для визуализации решающих деревьев
    '''
    x_min, x_max = data[:,0].min() - border, data[:,0].max() + border
    y_min, y_max = data[:,1].min() - border, data[:,1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

colors = ListedColormap(['red', 'blue', 'green'])
light_colors = ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])

def plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels,
                          colors = colors, lightcolors = light_colors):
    '''
    Процедура строит и рисует на графике визуальное представление решающего дерева
    примеры использований
    
    plot_decision_surface(tree.DecisionTreeClassifier(random_state = 1, max_depth = 3), train_data, train_labels, test_data, test_labels)
    
    plot_decision_surface(tree.DecisionTreeClassifier(random_state = 1, min_samples_leaf = 3), train_data, train_labels, test_data, test_labels)
    
    '''
    
    
    #fir model
    estimator.fit(train_data, train_labels)

    #set figure size
    plt.figure(figsize = (16,6))

    #plot decision surface on the train data
    plt.subplot(1,2,1)
    xx,yy = get_meshgrid(train_data)
    mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(train_data[:,0], train_data[:,1], c = train_labels, s = 100, cmap = colors)
    plt.title('Train data, accuracy=' + str(metrics.accuracy_score(train_labels, estimator.predict(train_data))))

    #plot decision surface on the test data
    plt.subplot(1,2,2)
    #xx,yy = get_meshgrid(train_data)
    #mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(test_data[:,0], test_data[:,1], c = test_labels, s = 100, cmap = colors)
    plt.title('Train data, accuracy=' + str(metrics.accuracy_score(test_labels, estimator.predict(test_data))))
    
    
def view_learning_curve(train_sizes, train_scores, test_scores):
    '''
    Функция строит кривую обучения по результатам выполнения learning_curve
    '''
    plt.grid(True)
    plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
    plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
    plt.ylim((0.0, 1.05))
    plt.legend(loc='lower right')
    
    
def view_learning_mass_cross_val_score(n_trees, scoring):
    plt.title("Accuracy score")
    plt.xlabel('n_trees')
    plt.ylabel('score')
    plt.plot(n_trees, scoring.mean(axis = 1), 'g-', marker='o', label='RandomForest')
    plt.grid(True)
    plt.legend(loc='lower right')    
    
    
def view_learn_losses(losses):  
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()    
    
    
def plot_trainig(train_losses, valid_losses, valid_accuracies):
    '''
    Выводит график обучения модели
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    info = fit(10, model, criterion, optimizer, *get_dataloaders(4))
    plot_trainig(*info)
    '''
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel("epoch")
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.xlabel("epoch")
    plt.plot(valid_accuracies, label="valid accuracy")
    plt.legend()
    
def imshow_from_dataloader(inp, title=None):
    '''
    Процедура выводит в ряд несколько картинок Dataloader
    # Получим 1 батч (картнки-метки) из обучающей выборки
    
    
    Пример использования
    inputs, classes = next(iter(dataloaders['train']))
    
    # Расположим картинки рядом
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])
    '''
    
    
    
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15, 12))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def view_batch():
    '''
    Пример выводит 9 картинок в матрице 3х3
    из датасета
    '''
    fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8), \
                            sharey=True, sharex=True)
    for fig_x in ax.flatten():
        random_characters = int(np.random.uniform(0,1000))
        im_val, label = datasets_list['train'][random_characters]
        img_label = " ".join(map(lambda x: x.capitalize(),\
                    datasets_list['train'].label_encoder.inverse_transform([label])[0].split('_')))
        imshow(im_val.data.cpu(), \
              title=img_label,plt_ax=fig_x)    