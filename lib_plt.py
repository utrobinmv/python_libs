import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from matplotlib.colors import ListedColormap

from sklearn import metrics

def plot_scater(x_coords, y_coords):
    '''
    Рисует точки на плоскости, с заданными координатами
    '''
    plt.scatter(x_coords, y_coords, color = 'r')

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
    
    