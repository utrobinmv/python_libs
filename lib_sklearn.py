import sklearn as sk
import math
import numpy as np

from sklearn import datasets, model_selection
from sklearn import linear_model, metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import tree

import xgboost as xgb

def make_circles(**args):
    '''
    params:
    noise - некий коэффициент шума от 0 до 1.

    Функция генерирует 2 окружности имеет другие параметры и 100 точек, при этом 
    Возврат:
    возвращает кортеж в котором во втором 1 или 0, 1 точка лежит на 1 окружности, 0 - лежит на другой окружности

    похожие функции: make_classification, make_regression, make_checkerboard, and etc


    '''
    return datasets.make_circles(**args)


def make_blobs(**args):
    '''
    Генерирует произвольную классификацию в виде облака точек классов
    datasets.make_blobs(centers=2, cluster_std=5.5, random_state=1)
    list_train, list_label = datasets.make_blobs(centers = 2, cluster_std = 5.5, random_state=1)
    '''
    return datasets.make_blobs(**args)

def make_classification(**args):
    '''
    params:
    n_classes - (4) количество классов
    n_features - общее количество признаков (2)
    n_informative - сколько из этих признаков является информативными (1)
    n_redundant - (1) - сколько из них является избыточными признаками
    n_repeated - количество дублированных признаков (0)
    n_clusters_per_class - (1) Количество кластеров в классе
    random_state - (1) ?

    до конца не понял, за что отвечает n_redundant

    Описание:
    Функция генерирует некую классификацию объектов (100 точек)

    Возвращает:
    кортеж из двух переменных: координаты точек, метки
    
    Пример исходный:
    classification_problem = datasets.make_classification(n_features = 2, n_informative = 2, n_classes = 3,
                                                      n_redundant = 0, n_clusters_per_class = 1, random_state = 1)

    '''

    return datasets.make_classification(**args)

def make_regression(**args):
    '''
    Генерирует произвольную линейную регрессию
    Может быть только одна линейная функция, остальные признаки это шум
    n_features = количество признаков
    noise - задает некий коэффициент неточности линейной регрессии
    n_informative = количество из них признаков не регрессии (шума)
    coef - венурть значение основного коэффициента уровнения линейной регрессии
    
    Вывод уровнения функции:
    print("y = {:.2f}*x1 + {:.2f}*x2".format(coef[0], coef[1], ...))
    '''
    #data, target, coef = datasets.make_regression(n_features = 2, n_informative = 1, n_targets = 1, 
    #                                              noise = 5., coef = True, random_state = 2)  
    
    
    
    return datasets.make_regression(**args)
    

def load_iris():
    '''
    Функция загружает предустановленный датасет 
    Возвращает:
    тип возвращаемого значениея sklearn.utils.Bunch
    по типу похож на словарь, с кучей данных
    iris.keys() - посмотреть ключи полученного словаря 
    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
    print(iris.DESCR) - получить описание датасета

    похожие функции: load_boston, load_diabetes, load_digits, load_linnerud, etc

    '''
    return datasets.load_iris()

#Модули кросс валидации

def train_test_split(data, target, **args):
    '''
    Разбивает data frame на две части тренировочную и тестовую выборку в заданной пропорции.
    возвращает кортеж из нескольких значений: train_data, test_data, train_label, test_label
    Параметры
     random_state
     shuffle - перемещать выборку
     train_size
     stratify
     
    '''
    model_selection.train_test_split()
    #model_selection.train_test_split(iris.data, iris.target, test_size = 0.3)
    #train_data, test_data, train_labels, test_labels = model_selection.t rain_test_split(list_train, list_label, test_size = 0.3, random_state = 1)
    return model_selection.train_test_split(data, target, **args)

def cross_validation_method(method, len_indicies, label_list, n_splits, shuffle = False, random_state = 1):
    x_data = range(0,len_indicies)
    if method == 1: #Метод KFold обычное разбиение, когда выборка делится на n_splits частей каждая часть с неким сдвигом
        #При этом возможно, что в тестовой выборке могут не попасть какие то классы, или наоборот
        kf = model_selection.KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
        for train_indices, test_indices in kf.split(x_data):
            print(train_indices, test_indices)        
    elif method == 2: #Самый нормальные, для деления выборок на n частей
        #Метод делит выборку так, чтобы метки каждого класса в тестовой выборке были (примерно) одинаковое количество раз
        kf = model_selection.StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
        for train_indices, test_indices in kf.split(x_data, label_list):
            print(train_indices, test_indices)        
        
def cross_validation_shuffle_method(method, len_indicies, label_list, n_splits, test_size = test_size):
    x_data = range(0,len_indicies)
    if method == 1: #Метод ShuffleSplit случайные выборки, один объект может 
        #делает из всей выборки n_splits случайных выборок, с заданной частью test_size
        #Но есть недостаток, дело в том, что в выборку тестовой могут попасть не все классы
        kf = model_selection.ShuffleSplit(n_splits = n_splits, test_size = test_size)
        for train_indices, test_indices in kf.split(x_data):
            print(train_indices, test_indices)        
    elif method == 2: #Самый нормальные, для произвольного количества выборок.
        #Метод генерит случайные выборки, при этом количество классов в тестовой выборке присутствуют все классы
        kf = model_selection.StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size)
        for train_indices, test_indices in kf.split(x_data, label_list):
            print(train_indices, test_indices)        
    elif method == 3:
        #Метод генерит столько выборок, сколько объектов в x_data, и где в тестовой выборке всегда только один элемент данных
        kf = model_selection.LeaveOneOut()
        for train_indices, test_indices in kf.split(x_data):
            print(train_indices, test_indices)
            
#Модули кросс валидации, для оценки качества моделей           

def model_cross_val_score(classifier, train_data, train_label):
    '''
    В функцию передается полный датасет данных и все метки классов.
    Метод формирует 10 выборок данных, и все их тренирует, далее выдает вероятность правильных ответов на каждой из 10 выборок
    Позволяет проверить среднюю точность данных на всем датасете
    '''
    model_scoring = model_selection.cross_val_score(classifier, train_data, train_label, scoring = 'accuracy', cv = 10)
    #model_scoring = model_selection.cross_val_score(classifier, train_data, train_label, scoring = 'mean_absolute_error', cv = 10)
    
    #Вывод результатов метрик качества тренировки на разных наборах выборки
    print('Log mean:{}, max:{}, min:{}, std:{}'.format(model_scoring.mean(), model_scoring.max(), 
                                                       model_scoring.min(), model_scoring.std()))

    return model_scoring
    
    
def model_cross_val_score_shuffle_metrics(classifier, train_data, train_label):
    '''
    Функция делает оценку модели, по произвольной функции (в данном случае accuracy_score)
    При этом сама делает разбивку тренировочных данных на 20 частей, в пропорции 70/30
    и после чего показывает результаты проведенной оценки
    '''
    
    scorer = metrics.make_scorer(metrics.accuracy_score)
    #scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=True) #Метрика greater_is_better говорит, что чем больше значение метрики тем лучше
    cv_strategy = model_selection.StratifiedShuffleSplit(n_splits=20, test_size = 0.3, random_state = 2)
    cv_strategy.get_n_splits(train_label)
    scoring = model_selection.cross_val_score(classifier, train_data, train_label, scoring = scorer, cv = cv_strategy)
    
    print('Ridge mean:{}, max:{}, min:{}, std:{}'.format(scoring.mean(), scoring.max(), 
                                                         scoring.min(), scoring.std()))
    

def model_cross_val_score_hand_made(train_data, train_label):
    '''
    Пример модели, когда перебирается конкретный параметр n_estimators, и далее проверяется его влияние на обучение модели
    
    результат можно посмотреть функцией view_learning_mass_cross_val_score(n_trees, scoring) из модуля lib_plt
    '''
    n_trees = [1] + list(range(10, 55, 5))
    scoring = []
    for n_tree in n_trees:
        estimator = ensemble.RandomForestClassifier(n_estimators = n_tree, min_samples_split=5)
        score = model_selection.cross_val_score(estimator, train_data, train_label, scoring = 'accuracy', cv = 5)
        scoring.append(score) #score - это 5 значений accuracy, на 5 выборках, так как параметр cv = 5
    scoring = np.asmatrix(scoring) 
    return n_trees, scoring
    
#Линейные модели

def classifier_ridge(train_data, train_labels, test_data):
    
    #Производит классификацию выборки с помощью классификации ridge
    
    #создание объекта - классификатора
    ridge_classifier = linear_model.RidgeClassifier(random_state = 1)
    
    #обучение классификатора
    ridge_classifier.fit(train_data, train_labels)

    print("Веса признаков: ", ridge_classifier.coef_)
    print("Коэффициент перед свободным членом (свободный коэффициент): ", ridge_classifier.intercept_)
    
    #применение обученного классификатора
    ridge_predictions = ridge_classifier.predict(test_data)
    
    return ridge_predictions

def classifier_coef(classifier):
    '''
    Функция возвращает коэффициенты обученной классификации регрессии.
    если результат вывести через print(), то можно сделать вывод какие из признаков дают веса для прогнозирования.
    Пример вывода:
         list(map(lambda x: round(x,2), regressor.coef_))
         [0.46, -0.45, 0.0, -0.01, 50.86, 148.01, -0.0, 0.01]
         
         в данном примере видно, что у признака 5 и 6 наибольшие веса, у остальных признаков веса близки к нолю. 
         это значит, что в данной выборке признаки 5 и 6 делают основной результат предсказания, а остальные параметры
         1, 2, 3, 4, 7, 8 - по сути в данном случае не требуются для прогнозирование. и вероятней всего их можно удалить из выборки данных
         без потери качества распознавания
         
    '''
    coefs = list(map(lambda x: round(x,2), classifier.coef_))
    return coefs

def classifier_model(model, train_data, train_labels, test_data):
    
    if model == 1:
        #Логистическая регрессия
        
        #создание объекта - классификатора
        classifier = linear_model.LogisticRegression(random_state = 1)

    elif model == 2:
        #Классификация ridge
        classifier = linear_model.RidgeClassifier(random_state = 1)
    elif model == 3:
        #Линейная регрессия
        classifier = linear_model.LinearRegression()
    elif model == 4:
        #Регрессия с использование регуляризации Лассо или регуляризации L1
        #В этой задаче происходит отбор признаков. Лассо регуляризация подходит для отбора признаков
        #Обучение на выборке с большим количеством признаков, система может показать, какие признаки являются значимыми, через функцию classifier.coef_
        #Если значение признака classifier.coef_ равно 0, то признак скорее всего незначим, и его можно отбросить
        classifier = linear_model.Lasso(random_state = 3)
    elif model == 5:
        #Модель классификации Стохатистический градиентный спуск
        # loss = используемая функция потерь:
        #                     функция потерь 'log' log loss - позволяет получить вероятностные результат        
        classifier = linear_model.SGDClassifier(loss = 'log', random_state = 1, max_iter=1000)
    elif model == 6:
        #Модель регрессии основанной на Стохатистическом градиентном спуске
        # loss = используемая функция потерь:
        #                     функция потерь 'log' log loss - позволяет получить вероятностные результат        
        classifier = linear_model.SGDRegressor(random_state = 1, max_iter = 20)
    elif model == 7:
        #Модель регрессии случайный лес
        classifier = ensemble.RandomForestRegressor(random_state = 0, max_depth = 20, n_estimators = 50)
    elif model == 8:
        #Модель решающих деревьев, пространство решений разделяет на площади, где каждую точку плоскости относит к определнному классу
        #max_depth - максимальное количество деревье (областей)
        classifier = tree.DecisionTreeClassifier(random_state = 1, max_depth = 3)
        #classifier = tree.DecisionTreeClassifier(random_state = 1, min_samples_leaf = 3)
    elif model == 9:
        classifier = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 2, random_state = 1)
        #classifier = ensemble.RandomForestClassifier(n_estimators = 40, min_samples_split=5)
    elif model == 10:
        classifier = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 1)
    elif model == 10:
        classifier = ensemble.GradientBoostingRegressor(n_estimators = 50, random_state = 1)
    elif model == 11:
        #Модель XGB Boost
        classifier = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators = 40, min_child_weight=3)
        
        
  
    #обучение классификатора
    classifier.fit(train_data, train_labels)

    print("Веса признаков: ", classifier.coef_)
    print("Коэффициент перед свободным членом (свободный коэффициент): ", classifier.intercept_)
    
    #применение обученного классификатора
    predictions = classifier.predict(test_data)
    
    proba_predictions = []
      
    if model == 1:
        #Логистическая регрессия - получение вероятности принадлежности каждому классу
        proba_predictions = classifier.predict_proba(test_data)
    elif model == 2:
        #Не предоставляет вероятности
        pass
    elif model == 3:
        #Не предоставляет вероятности
        pass
    elif model == 4:
        #Не предоставляет вероятности
        pass
    elif model == 5:
        #Не предоставляет вероятности
        proba_predictions = classifier.predict_proba(test_data)
    elif model == 6:
        #Не предоставляет вероятности
        pass
      
    return classifier, predictions, proba_predictions


#Оценка качества моделей

def metrics_accuracy_score(test_labels, labels_predict):
    '''
    Метрика: количество правильных ответов
    '''
    return metrics.accuracy_score(test_labels, labels_predict)

def all_metrics_score(metric, test_labels, labels_predict):
    '''
    Различные варианты метрик качества выборки
    '''
    if metric == 1: #Классификация
        #Метрика accuracy_score - количество правильных ответов. Показывает долю правильных ответов
        #Код функции accuracy_score
        #sum([1. if pair[0] == pair[1] else 0. for pair in zip(test_labels, labels_predict)])/len(test_labels)
        return metrics.accuracy_score(test_labels, labels_predict)
    elif metric == 2: #Регрессия
        #Метрика mean_absolute_error - среднее отклонение (Средняя ошибка предсказаний)
        return metrics.mean_absolute_error(test_labels, labels_predict)
    elif metric == 3: #Классификация
        #Метрика precision_score - оценка точности классификации. Как часта алгоритм предсказал и оказался прав.
        #Это метрика False Positive
        #return metrics.precision_score(test_labels, labels_predict, pos_label = 0) #Точность классификации к нулевому классу
        return metrics.precision_score(test_labels, labels_predict) #default  pos_label = 1, Точность классификации к первому классу
    elif metric == 4: #Классификация
        #Метрика recall_score - оценка полноты классификации. Сколько объектов нашел алгоритм
        #Это метрика False Negative
        #return metrics.recall_score(test_labels, labels_predict, pos_label = 0) #Точность классификации к нулевому классу
        return metrics.recall_score(test_labels, labels_predict) #default  pos_label = 1, Точность классификации к первому классу
    elif metric == 5: #Классификация
        #Метрика f1_score - оценка f меры
        #Метрика является гибридом двух других, precision_score и recall_score
        '''
        Очевидный недостаток пары метрик precision-recall - в том, что их две: непонятно,
        как ранжировать алгоритмы. Чтобы этого избежать, используют F1-метрику,
        которая равна среднему гармоническому precision и recall.
        F1-метрика будет равна 1, если и только если precision = 1 и recall = 1 (идеальный алгоритм).
        
        (: Обмануть F1 сложно: если одна из величин маленькая, а другая близка к 1 (по графикам видно,
        что такое соотношение иногда легко получить), F1 будет далека от 1. F1-метрику сложно оптимизировать,
        потому что для этого нужно добиваться высокой полноты и точности одновременно.

        Например, посчитаем F1 для того же набора векторов, для которого мы строили графики (мы помним,
        что там одна из кривых быстро выходит в единицу).
        '''
        #return metrics.f1_score(test_labels, labels_predict, pos_label = 0) #Точность классификации к нулевому классу
        return metrics.f1_score(test_labels, labels_predict) #default  pos_label = 1, Точность классификации к первому классу
    elif metric == 6: #Классификация
        #Метрика Площадь рок кривой
        #return metrics.roc_auc_score(clf_test_labels, probability_predictions[:,1]) #В метрику можно передавать не только метки классов, но и вероятности результатов, результаты функции predict_proba
        return metrics.roc_auc_score(test_labels, labels_predict)
    elif metric == 7: #Классификация
        #Метрика PR AUC
        return metrics.average_precision_score(test_labels, labels_predict)
    elif metric == 8: #Классификация
        #Метрика log_loss Логистических потерь (Оценка вероятностных классификаторов)
        #Передаются только вероятности классов, результаты функции predict_proba
        #Чем меньше значение log_loss - тем лучше
        #return metrics.log_loss(clf_test_labels, probability_predictions[:,1]) #В качестве второго параметра переданы вероятности принадлежности к первому классу
        pass
    elif metric == 9: #Регрессия
        #Метрика mean_absolute_error Обычное отклонение
        return metrics.mean_absolute_error(test_labels, labels_predict)
    elif metric == 10: #Регрессия
        #Метрика mean_squared_error Средне квадратичное отклонение
        return metrics.mean_squared_error(test_labels, labels_predict)
    elif metric == 11: #Регрессия
        #Метрика sqrt_mean_squared_error Корень из средне квадратичного отклонения
        return math.sqrt(metrics.mean_squared_error(test_labels, labels_predict))
    elif metric == 11: #Регрессия
        #Метрика r2_score Коэффициент детерминации
        return metrics.r2_score(test_labels, labels_predict)


#Подбор оптимальных параметров

def find_model_optimized_parameters(classifier, parameters_grid, train_data, train_label):
    '''
    Функция подбирает оптимальные параметры модели
    Входные данные:
    classifier - модель классификации
    train_data - данные обучения
    train_label - метки обучения
    
    Пример, 
       classifier = linear_model.SGDClassifier(random_state = 0)
       parameters_grid = {
              'loss' : ['hinge', 'log', 'squared_hinge', 'squared_loss'],
              'penalty' : ['l1', 'l2'],
              'n_iter_no_change' : range(5,10),
              'alpha' : np.linspace(0.0001, 0.001, num=5),
              }
    '''
    
    print(classifier.get_params().keys()) #Выводим список всех возможных параметров модели
    '''
    Пример ответа:
    dict_keys(['alpha', 'average', 'class_weight', 'early_stopping', 'epsilon', 'eta0', 'fit_intercept', 'l1_ratio',
    'learning_rate', 'loss', 'max_iter', 'n_iter_no_change', 'n_jobs', 'penalty', 'power_t', 'random_state',
    'shuffle', 'tol', 'validation_fraction', 'verbose', 'warm_start'])
    '''
    
    cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size = 0.2, random_state = 0) #Будет сгенерировано 10 случайных выборок, из train_data
    
    grid_cv = GridSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv)
    
    grid_cv.fit(train_data, train_label)
    
    print(grid_cv) #Информация
    '''
    Пример ответа:
    GridSearchCV(cv=StratifiedShuffleSplit(n_splits=10, random_state=0, test_size=0.2,
            train_size=None),
             estimator=SGDClassifier(random_state=0),
             param_grid={'alpha': array([0.0001  , 0.000325, 0.00055 , 0.000775, 0.001   ]),
                         'loss': ['hinge', 'log', 'squared_hinge',
                                  'squared_loss'],
                         'n_iter_no_change': range(5, 10),
                         'penalty': ['l1', 'l2']},
             scoring='accuracy')
    '''
    
    print(grid_cv.best_score_) #Лучший результат
    print(grid_cv.best_params_) #Список лучших значений параметров
    '''
    Пример ответа:
    parameters_grid = {
      'loss' : ['hinge', 'log', 'squared_hinge', 'squared_loss'],
      'penalty' : ['l1', 'l2'],
      'n_iter_no_change' : range(5,10),
      'alpha' : np.linspace(0.0001, 0.001, num=5),
    }
    '''
    
    return grid_cv.best_params_


def find_model_random_optimized_parameters(classifier, parameters_grid, train_data, train_label, n_iter):
    '''
    Функция подбирает оптимальные параметры модели
    Аналогично функции выше find_model_optimized_parameters, однако при большом количестве данных,
    или большом количестве параметров позволяет выполнить поиск быстрее, 
    за счет заданного количество произвольных случайных подборов параметров
    вместо перебора всех возможных комбинаций
    '''
    cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size = 0.2, random_state = 0) #Будет сгенерировано 10 случайных выборок, из train_data
    random_grid_cv = RandomizedSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv, n_iter=n_iter, random_state=0) #n_iter - количество случайных итераций
    random_grid_cv.fit(train_data, train_label)
    return random_grid_cv.best_params_



        
#Визуальные метрики       

def classification_report(test_labels, labels_predict):
    '''
    Функция выводит некий отчет сразу по нескольким метрикам качества: precision, recall, f1-score, accuracy
    используется если необходимо визуально оценить качество сразу по нескольким метрикам
    
    '''
    print(metrics.classification_report(test_labels, labels_predict))
        
def metrics_roc_curve(clf_test_labels, probability_predictions):
    '''
    Метрика, под названием Рок кривая. Чем больше площадь рок кривой, тем больше доля правильных ответов.
    Данная процедура позволяет нарисовать и увидеть эту roc_curve
    '''
    '''
    fpr, tpr, _ = metrics.roc_curve(clf_test_labels, probability_predictions[:,1])
    pylab.plot(fpr, tpr, label = 'linear model')
    pylab.plot([0, 1], [0, 1], '--', color = 'grey', label = 'random')
    pylab.xlim([-0.05, 1.05])
    pylab.ylim([-0.05, 1.05])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve')
    pylab.legend(loc = "lower right")
    '''
    pass
    
    
def metrics_precision_recall_curve(actual, predicted):
    '''
    Функция строит график сходимости метрик precision и recall
    позволяет выявить оптимальный порог отсечки правильных и неправильных ответов
    При увеличении порога мы делаем меньше ошибок FP и больше ошибок FN,
    поэтому одна из кривых растет, а вторая  - падает.
    По такому графику можно подобрать оптимальное значение порога,
    при котором precision и recall будут приемлемы. Если такого порога не нашлось,
    нужно обучать другой алгоритм.
    '''
    prec, rec, thresh = precision_recall_curve(actual, predicted)
    
    '''
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 3, 1)
    plt.plot(thresh, prec[:-1], label="precision")
    plt.plot(thresh, rec[:-1], label="recall")
    plt.xlabel("threshold")
    ax.set_title('Typical')
    plt.legend()    
    '''

def confusion_matrix(test_labels, labels_predict):
    '''
    Построение матрицы точности объекта (confusion matrix)
    Матрица размером, количество классов, на количество классов.
    
    На диагонали находится количество объектов, на которых мы ответили правильно
    Вне диагонали, количество объектов на которых мы ошиблись.
    '''
    matrix = metrics.confusion_matrix(test_labels, labels_predict)
    print(matrix)
    
    #Количество правильных ответом можно посчитать вот так
    #Формула sum([1 if pair[0] == pair[1] else 0 for pair in zip(test_labels, labels_predict)])
    
    verno_kolvo = matrix.diagonal().sum()
    
    return verno_kolvo #Количество правильных ответов

    

#Масштабирование признаков данных
def data_scaler(train_data, test_data, train_labels):
    '''
    Функция выполняет масштабирование признаков в наборе данных, чтобы по нему можно было проводить последующее обучение
    Возвращает: отмасштабированные данные scaler_train_data и scaled_test_data
    '''
    scaler = StandardScaler()
    scaler.fit(train_data, train_labels)
    scaler_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    return scaler_train_data, scaled_test_data


#Pipeline - возможность сразу обучать последовательно несколькими алгоритмами

def pipeline_fit_scaler_model(train_data, train_labels, test_data, test_labels):
    '''
    Функция производит нормализацию данных, а затем поиск функции регрессии
    т.е. в рамках одного обучения, по сути выполняется сразу два обучения
    '''
    scaler = StandardScaler()
    regressor = linear_model.SGDRegressor(random_state = 0)
    pipeline = Pipeline(steps = [('scaling', scaler),('regression', regressor)])
    pipeline.fit(train_data, train_labels)
    print(metrics.mean_absolute_error(test_labels, pipeline.predict(test_data))) #Метрика результата обучения
    
    return pipeline


def pipeline_GridSearch_fit_scaler_model(train_data, train_labels, test_data, test_labels):
    '''
    Функция производит оптимальный подбор параметров для salied данных
    '''
    scaler = StandardScaler()
    regressor = linear_model.SGDRegressor(random_state = 0)
    pipeline = Pipeline(steps = [('scaling', scaler),('regression', regressor)])
    print(pipeline.get_params().keys()) #Список возможных параметров оптимизации
    
    parameters_grid = {
        'regression__loss' : ['epsilon_insensitive', 'squared_loss'],
        'regression__n_iter_no_change' : [3,5,10],
        'regression__penalty' : ['l1', 'l2', 'none'],
        'regression__alpha' : [0.0001, 0.01],
        'scaling__with_mean' : [0, 1],
    }
    
    grid_cv = GridSearchCV(pipeline, parameters_grid, scoring = 'neg_mean_absolute_error', cv = 4)
    grid_cv.fit(train_data, train_labels)
    return grid_cv.best_params_

def pipeline_sorting_grid_and_learn(regressor, train_data, train_labels, binary_data_columns, categorical_data_columns, numeric_data_columns):
    '''
    Данная функция преобразовывает входной pandas dataframe, бинарные признаки, категориальные признаки и номерные, в общий дата сет
    по которому далее обучает модель regressor
    
    Входные данные
    regressor - модель прогнозирования
    binary_data_columns - список колонок из дата фрейма относящиеся к данному типу
    
    примеры входных данных
    regressor = linear_model.SGDRegressor(random_state = 1, max_iter = 20)
    regressor = linear_model.SGDRegressor(random_state = 0, max_iter = 3, loss = 'squared_loss', penalty = 'l2')
    categorical_data_columns = ['season', 'weather', 'month']
    binary_data_columns = ['holiday', 'workingday']
    numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
    
    '''
    
    binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)
    categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)
    numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)
    
    estimator = Pipeline(steps = [
        ('feature_processing', FeatureUnion(transformer_list = [
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data.iloc[:, binary_data_indices])),
    
            #numeric
            ('numeric_variables_processing', Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0.))
            ])),
            
            #categorical
            ('categorical_variables_processing', Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))
            ])),
        ])),
        ('model_fitting', regressor)
        ]
        )
    
    estimator.fit(train_data, train_labels)
    
    #Бонусный блок, проводит оптимизацию по параметрам
    
    #estimator.get_params().keys() - список параметров для оптимизации
    #parameters_grid = {
    #        'model_fitting__alpha' : [0.0001, 0.001, 0.1],
    #        'model_fitting__eta0' : [0.001, 0.05],
    #}
    #grid_cv = GridSearchCV(estimator, parameters_grid, scoring = 'neg_mean_absolute_error', cv = 4)
    #grid_cv.fit(train_data, train_labels)
    #print(grid_cv.best_score_)
    #print(grid_cv.best_params_)
    
    return estimator


#Кривая обучения

def find_learning_curve(estimator, train_data, train_label):
    '''
    Данная функция определяет требуется ли увеличивать размер обучающей выборки
    для повышения качества модели estimator
    
    
    Результат модели можно посмотреть с помощью функции из lib_plt - view_learning_curve
    '''
    
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator, train_data, train_label, 
                                                                           train_sizes=np.arange(0.1,1., 0.2), 
                                                                           cv=3, scoring='accuracy')
    
    
    #print(train_sizes)
    #print(train_scores.mean(axis = 1))
    #print(test_scores.mean(axis = 1))    
    
    return train_sizes, train_scores, test_scores


